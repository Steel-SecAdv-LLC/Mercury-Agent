# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid Quantum-Classical Optimization for Mercury Agent.

Implements VQE, QAOA, and quantum kernel methods for anomaly detection.

References:
- Peruzzo et al. (2014): A variational eigenvalue solver on a photonic quantum processor
- Farhi et al. (2014): A Quantum Approximate Optimization Algorithm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.quantum_computing.circuits import (
    AnomalyEncodingCircuit,
    EncodingType,
    QuantumCircuitBuilder,
    QuantumFeatureMap,
    SimulatedQuantumCircuit,
    VariationalAnsatz,
    VariationalCircuit,
)
from omni_mercury_engine.quantum_computing.executor import (
    QuantumExecutor,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hybrid quantum-classical optimization."""

    optimal_parameters: np.ndarray[Any, Any]
    optimal_value: float
    n_iterations: int
    convergence_history: list[float]
    final_circuit: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class ClassicalOptimizer:
    """Classical optimizer for variational parameter updates.

    Implements gradient-free and gradient-based optimization methods.
    """

    def __init__(
        self,
        method: str = "cobyla",
        maxiter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            method: Optimization method.
            maxiter: Maximum iterations.
            tol: Convergence tolerance.
            seed: Optional seed for the per-instance ``Generator`` driving
                SPSA's ±1 perturbation direction. ``None`` (default) uses
                OS entropy.
        """
        self._method = method.lower()
        self._maxiter = maxiter
        self._tol = tol
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def minimize(
        self,
        objective: Callable[[np.ndarray[Any, Any]], float],
        initial_params: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float, list[float]]:
        """Minimize the objective function.

        Args:
            objective: Function to minimize
            initial_params: Initial parameter values

        Returns:
            Tuple of (optimal_params, optimal_value, history)
        """
        if self._method == "cobyla":
            return self._cobyla_minimize(objective, initial_params)
        elif self._method == "coordinate_search":
            return self._coordinate_search(objective, initial_params)
        elif self._method == "spsa":
            return self._spsa_minimize(objective, initial_params)
        elif self._method == "gradient_descent":
            return self._gradient_descent(objective, initial_params)
        else:
            return self._cobyla_minimize(objective, initial_params)

    def _cobyla_minimize(
        self,
        objective: Callable[[np.ndarray[Any, Any]], float],
        initial_params: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float, list[float]]:
        """COBYLA (Powell 1994) via SciPy, gradient-free.

        ``maxiter`` bounds the objective evaluations (at least the
        ``n + 2`` COBYLA needs to build its first simplex); ``history``
        records every evaluated value and ends with the value at the
        returned parameters. (This method ran a coordinate search under the
        COBYLA name; that search is ``method="coordinate_search"``.)
        """
        from scipy.optimize import minimize

        history: list[float] = []

        def recorded(params: np.ndarray[Any, Any]) -> float:
            value = float(objective(params))
            history.append(value)
            return value

        start = np.asarray(initial_params, dtype=float)
        if self._maxiter == 0 or start.size == 0:
            # Nothing to optimise: the objective at the start, no history.
            return start.copy(), float(objective(start)), history
        result = minimize(
            recorded,
            start,
            method="COBYLA",
            options={
                "maxiter": max(self._maxiter, start.size + 2),
                "rhobeg": 1.0,
                "tol": self._tol,
            },
        )
        params = np.asarray(result.x, dtype=float)
        value = recorded(params)
        return params, value, history

    def _coordinate_search(
        self,
        objective: Callable[[np.ndarray[Any, Any]], float],
        initial_params: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float, list[float]]:
        """Coordinate pattern search with a shrinking step (gradient-free).

        ``history`` ends with the value at the returned parameters: when the
        last sweep moved them, its value is appended (it was dropped, so the
        returned value predated the final sweep).
        """
        params = initial_params.copy()
        history = []
        rho = 1.0
        current_value = 0.0
        evaluated = params.copy()

        for iteration in range(self._maxiter):
            current_value = objective(params)
            history.append(current_value)
            evaluated = params.copy()

            if iteration > 0 and abs(history[-1] - history[-2]) < self._tol:
                break

            for i in range(len(params)):
                trial_params = params.copy()
                trial_params[i] += rho

                trial_value = objective(trial_params)
                if trial_value < current_value:
                    params = trial_params
                    current_value = trial_value
                else:
                    trial_params[i] = params[i] - rho
                    trial_value = objective(trial_params)
                    if trial_value < current_value:
                        params = trial_params
                        current_value = trial_value

            rho *= 0.95

        if history and not np.array_equal(params, evaluated):
            history.append(current_value)
        return params, history[-1] if history else objective(params), history

    def _spsa_minimize(
        self,
        objective: Callable[[np.ndarray[Any, Any]], float],
        initial_params: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float, list[float]]:
        """SPSA optimization (stochastic gradient approximation)."""
        params = initial_params.copy()
        history = []

        a = 0.1
        c = 0.1
        A = self._maxiter // 10
        alpha = 0.602
        gamma = 0.101

        for k in range(self._maxiter):
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma

            delta = self._rng.choice([-1, 1], size=len(params))

            f_plus = objective(params + ck * delta)
            f_minus = objective(params - ck * delta)

            gradient_approx = (f_plus - f_minus) / (2 * ck * delta + 1e-10)
            params = params - ak * gradient_approx

            current_value = objective(params)
            history.append(current_value)

            if k > 0 and abs(history[-1] - history[-2]) < self._tol:
                break

        return params, history[-1] if history else objective(params), history

    def _gradient_descent(
        self,
        objective: Callable[[np.ndarray[Any, Any]], float],
        initial_params: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float, list[float]]:
        """Gradient descent with finite differences.

        ``history`` ends with the value at the returned parameters: when the
        last step moved them, their value is appended (the update was not
        evaluated, so the returned value was the objective one step earlier).
        """
        params = initial_params.copy()
        history = []
        learning_rate = 0.1
        epsilon = 0.01

        evaluated = params.copy()
        for iteration in range(self._maxiter):
            current_value = objective(params)
            history.append(current_value)
            evaluated = params.copy()

            gradient = np.zeros_like(params)
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += epsilon
                gradient[i] = (objective(params_plus) - current_value) / epsilon

            params = params - learning_rate * gradient
            learning_rate *= 0.99

            if iteration > 0 and abs(history[-1] - history[-2]) < self._tol:
                break

        if history and not np.array_equal(params, evaluated):
            history.append(objective(params))
        return params, history[-1] if history else objective(params), history


class HybridOptimizer:
    """Hybrid quantum-classical optimizer.

    Combines variational quantum circuits with classical optimization.
    """

    def __init__(
        self,
        executor: QuantumExecutor | None = None,
        optimizer_method: str = "cobyla",
        maxiter: int = 100,
        seed: int | None = None,
    ) -> None:
        """Initialize the hybrid optimizer.

        Args:
            executor: Optional quantum executor.
            optimizer_method: Inner classical optimizer method.
            maxiter: Maximum iterations.
            seed: Optional seed for the per-instance ``Generator`` driving
                random initial-parameter generation when the caller does
                not supply ``initial_params``.  ``None`` (default) uses OS
                entropy.  Threaded into the inner ``ClassicalOptimizer``
                so SPSA perturbations are also reproducible under the
                same seed.
        """
        if executor is None:
            executor = QuantumExecutor()
        self._executor = executor
        self._optimizer = ClassicalOptimizer(optimizer_method, maxiter, seed=seed)
        self._builder = QuantumCircuitBuilder()
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def optimize(
        self,
        variational_circuit: VariationalCircuit,
        cost_function: Callable[[dict[str, int]], float],
        initial_params: np.ndarray[Any, Any] | None = None,
    ) -> OptimizationResult:
        """Optimize variational circuit parameters.

        Args:
            variational_circuit: Variational circuit to optimize
            cost_function: Function mapping measurement counts to cost
            initial_params: Initial parameters (random if None)

        Returns:
            OptimizationResult with optimal parameters
        """
        if initial_params is None:
            initial_params = self._rng.uniform(0, 2 * np.pi, variational_circuit.num_parameters)

        def objective(params: np.ndarray[Any, Any]) -> float:
            circuit = variational_circuit.build(params)
            circuit.measure_all()
            result = self._executor.run(circuit)

            if isinstance(result, list):
                result = result[0]

            return cost_function(result.counts)

        optimal_params, optimal_value, history = self._optimizer.minimize(objective, initial_params)

        final_circuit = variational_circuit.build(optimal_params)

        return OptimizationResult(
            optimal_parameters=optimal_params,
            optimal_value=optimal_value,
            n_iterations=len(history),
            convergence_history=history,
            final_circuit=final_circuit,
        )


class QuantumKernel:
    """Quantum kernel for kernel-based machine learning.

    Computes kernel matrix using quantum feature maps.
    """

    def __init__(
        self,
        num_qubits: int,
        feature_map_reps: int = 2,
        executor: QuantumExecutor | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the quantum kernel; ``seed`` seeds the measurement shots."""
        self._feature_map = QuantumFeatureMap(num_qubits, feature_map_reps, seed=seed)
        self._executor = executor or QuantumExecutor()
        self._num_qubits = num_qubits

    def compute_kernel_matrix(
        self,
        X: np.ndarray[Any, Any],
        Y: np.ndarray[Any, Any] | None = None,
        shots: int | None = 1024,
    ) -> np.ndarray[Any, Any]:
        """Compute quantum kernel matrix.

        Args:
            X: First data matrix (n_samples, n_features)
            Y: Second data matrix (default: X)
            shots: Number of shots for kernel estimation, or ``None`` for the
                exact fidelity on the built-in simulator

        Returns:
            Kernel matrix K[i,j] = k(X[i], Y[j])
        """
        if Y is None:
            Y = X

        n_x = X.shape[0]
        n_y = Y.shape[0]
        K = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):
                K[i, j] = self._feature_map.compute_kernel(X[i], Y[j], shots)

        return K

    def fit_svm(
        self,
        X_train: np.ndarray[Any, Any],
        y_train: np.ndarray[Any, Any],
        C: float = 1.0,
    ) -> Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
        """Fit quantum kernel SVM.

        Args:
            X_train: Training features
            y_train: Training labels
            C: Regularization parameter

        Returns:
            Prediction function
        """
        K = self.compute_kernel_matrix(X_train)

        n = len(y_train)
        alpha = np.zeros(n)
        learning_rate = 0.01

        for _ in range(100):
            for i in range(n):
                margin = y_train[i] * np.sum(alpha * y_train * K[i, :])
                if margin < 1:
                    alpha[i] += learning_rate * (1 - margin)
                alpha[i] = np.clip(alpha[i], 0, C)

        support_vectors = alpha > 1e-5
        self._sv_X = X_train[support_vectors]
        self._sv_y = y_train[support_vectors]
        self._sv_alpha = alpha[support_vectors]

        def predict(X_test: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            K_test = self.compute_kernel_matrix(X_test, self._sv_X)
            predictions = np.sign(np.sum(self._sv_alpha * self._sv_y * K_test, axis=1))
            return predictions

        return predict


class VQEAnomalyDetector:
    """Variational one-class anomaly detector.

    The variational circuit is trained so that, after each normal training
    sample is encoded, it maps the state to ``|0...0>``: the loss is the
    mean of ``1 - P(|0...0>)`` over the training samples, the compression
    objective of the quantum autoencoder (Romero, Olson & Aspuru-Guzik
    2017, Quantum Sci. Technol. 2, 045001). A sample's anomaly score is the
    same ``1 - P(|0...0>)``: samples like the training data compress, others
    do not. Every feature is encoded (:meth:`AnomalyEncodingCircuit.encode_layered`).
    On a simulator the probabilities are exact. An executor that returns
    measured counts is scored from them; the hardware backend submits jobs
    asynchronously and returns none from ``run``, so fitting or scoring on
    it raises ``NotImplementedError``. (The former fit computed the training mean, discarded
    it, and minimised the parity of the bare circuit, so its parameters
    never saw the data and held-out AUROC sat at chance.)
    """

    #: Training samples used per loss evaluation (a fixed, seeded subset).
    MAX_TRAINING_BATCH = 64

    def __init__(
        self,
        num_qubits: int,
        ansatz: VariationalAnsatz = VariationalAnsatz.REAL_AMPLITUDES,
        reps: int = 2,
        executor: QuantumExecutor | None = None,
        seed: int | None = 0,
    ) -> None:
        """Initialize the VQE detector."""
        self._num_qubits = num_qubits
        self._variational = VariationalCircuit(num_qubits, ansatz, reps)
        self._optimizer = HybridOptimizer(executor, seed=seed)
        self._encoding = AnomalyEncodingCircuit(num_qubits, EncodingType.ANGLE)
        self._optimal_params: np.ndarray[Any, Any] | None = None
        self._training_energy = 0.0
        self._rng = np.random.default_rng(seed)

    def fit(
        self,
        X_train: np.ndarray[Any, Any],
        maxiter: int = 50,
    ) -> VQEAnomalyDetector:
        """Train VQE anomaly detector on normal data.

        Args:
            X_train: Training data (normal samples), scaled to [0, 1]
            maxiter: Maximum objective evaluations (COBYLA)

        Returns:
            self for method chaining

        Raises:
            ValueError: If the training data is empty or not finite.
        """
        X = np.atleast_2d(np.asarray(X_train, dtype=float))
        if X.size == 0 or not np.isfinite(X).all():
            raise ValueError("VQE training data must be non-empty and finite")
        if X.shape[0] > self.MAX_TRAINING_BATCH:
            X = X[self._rng.choice(X.shape[0], self.MAX_TRAINING_BATCH, replace=False)]
        encoded = [self._encoding.encode_layered(sample) for sample in X]

        def loss(params: np.ndarray[Any, Any]) -> float:
            ansatz = self._variational.build(params)
            return float(np.mean([1.0 - self._zero_probability(enc, ansatz) for enc in encoded]))

        optimizer = ClassicalOptimizer("cobyla", maxiter=maxiter)
        initial = self._rng.uniform(-0.1, 0.1, self._variational.num_parameters)
        params, value, _history = optimizer.minimize(loss, initial)

        self._optimal_params = params
        self._training_energy = value

        return self

    def score(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute anomaly scores for samples.

        Args:
            X: Data to score, scaled with the training statistics

        Returns:
            Anomaly scores ``1 - P(|0...0>)`` in [0, 1] (higher = more anomalous)
        """
        if self._optimal_params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        ansatz = self._variational.build(self._optimal_params)
        return np.array(
            [
                1.0 - self._zero_probability(self._encoding.encode_layered(sample), ansatz)
                for sample in np.atleast_2d(np.asarray(X, dtype=float))
            ]
        )

    def _compose(self, encoding_circuit: Any, variational_circuit: Any) -> Any:
        """Encoding followed by the variational circuit."""
        if isinstance(encoding_circuit, SimulatedQuantumCircuit):
            combined = QuantumCircuitBuilder().create_circuit(self._num_qubits)
            for gate, qubits, params in encoding_circuit._gates:
                self._apply_gate(combined, gate, qubits, params)
            for gate, qubits, params in variational_circuit._gates:
                self._apply_gate(combined, gate, qubits, params)
            return combined
        return encoding_circuit.compose(variational_circuit)

    def _zero_probability(self, encoding_circuit: Any, variational_circuit: Any) -> float:
        """``P(|0...0>)`` after encoding and the variational circuit."""
        combined = self._compose(encoding_circuit, variational_circuit)
        executor = self._optimizer._executor
        if getattr(executor, "is_simulator", False):
            if isinstance(combined, SimulatedQuantumCircuit):
                return float(combined.probabilities()[0])
            from qiskit.quantum_info import Statevector

            return float(Statevector(combined).probabilities()[0])
        combined.measure_all()
        result = executor.run(combined)
        if isinstance(result, list):
            if not result:
                raise NotImplementedError(
                    "VQE evaluates each circuit synchronously, but this backend runs "
                    "jobs asynchronously and returned no result; use a simulator"
                )
            result = result[0]
        total = sum(result.counts.values())
        return result.counts.get("0" * self._num_qubits, 0) / total if total else 0.0

    def _apply_gate(
        self,
        circuit: SimulatedQuantumCircuit,
        gate: str,
        qubits: list[int],
        params: list[float],
    ) -> None:
        """Apply gate to circuit."""
        if gate == "h":
            circuit.h(qubits[0])
        elif gate == "x":
            circuit.x(qubits[0])
        elif gate == "rx":
            circuit.rx(params[0], qubits[0])
        elif gate == "ry":
            circuit.ry(params[0], qubits[0])
        elif gate == "rz":
            circuit.rz(params[0], qubits[0])
        elif gate == "cx":
            circuit.cx(qubits[0], qubits[1])
        elif gate == "cz":
            circuit.cz(qubits[0], qubits[1])

    def _compute_energy(self, counts: dict[str, int]) -> float:
        """Compute energy from measurement counts."""
        total = sum(counts.values())
        energy = 0.0
        for bitstring, count in counts.items():
            parity = sum(int(b) for b in bitstring) % 2
            energy += parity * count / total
        return energy


class QAOAAnomalyDetector:
    """Quantum Approximate Optimization Algorithm for anomaly detection.

    Uses QAOA to solve combinatorial anomaly problems.
    """

    def __init__(
        self,
        num_qubits: int,
        p: int = 2,
        executor: QuantumExecutor | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the QAOA detector.

        Args:
            num_qubits: Number of qubits.
            p: QAOA layer count.
            executor: Optional quantum executor.
            seed: Optional seed for the per-instance ``Generator`` driving
                random initial-parameter generation in ``optimize``.
                ``None`` (default) uses OS entropy.
        """
        self._num_qubits = num_qubits
        self._p = p
        self._executor = executor or QuantumExecutor()
        self._optimal_params: np.ndarray[Any, Any] | None = None
        self._builder = QuantumCircuitBuilder()
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def build_qaoa_circuit(
        self,
        gamma: list[float],
        beta: list[float],
        cost_terms: list[tuple[int, int, float]],
    ) -> Any:
        """Build QAOA circuit.

        Args:
            gamma: Mixer angles
            beta: Cost angles
            cost_terms: List of (i, j, weight) for ZZ interactions

        Returns:
            QAOA circuit
        """
        circuit = self._builder.create_circuit(self._num_qubits)

        for i in range(self._num_qubits):
            circuit.h(i)

        for layer in range(self._p):
            for i, j, weight in cost_terms:
                circuit.cx(i, j)
                circuit.rz(2 * gamma[layer] * weight, j)
                circuit.cx(i, j)

            for i in range(self._num_qubits):
                circuit.rx(2 * beta[layer], i)

        return circuit

    def fit(
        self,
        adjacency_matrix: np.ndarray[Any, Any],
        maxiter: int = 50,
    ) -> QAOAAnomalyDetector:
        """Train QAOA on graph-based anomaly problem.

        Args:
            adjacency_matrix: Graph adjacency matrix
            maxiter: Maximum optimization iterations

        Returns:
            self for method chaining
        """
        cost_terms = []
        for i in range(self._num_qubits):
            for j in range(i + 1, min(self._num_qubits, adjacency_matrix.shape[0])):
                if adjacency_matrix[i, j] != 0:
                    cost_terms.append((i, j, adjacency_matrix[i, j]))

        self._cost_terms = cost_terms

        initial_params = self._rng.uniform(0, np.pi, 2 * self._p)

        def objective(params: np.ndarray[Any, Any]) -> float:
            gamma = list(params[: self._p])
            beta = list(params[self._p :])

            circuit = self.build_qaoa_circuit(gamma, beta, cost_terms)
            circuit.measure_all()

            result = self._executor.run(circuit)
            if isinstance(result, list):
                result = result[0]

            return -self._compute_cost(result.counts, cost_terms)

        optimizer = ClassicalOptimizer("cobyla", maxiter)
        self._optimal_params, _, _ = optimizer.minimize(objective, initial_params)

        return self

    def score(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute anomaly scores using QAOA.

        Args:
            X: Data points to score

        Returns:
            Anomaly scores
        """
        if self._optimal_params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        scores = []

        gamma = list(self._optimal_params[: self._p])
        beta = list(self._optimal_params[self._p :])

        for sample in X:
            cost_terms = [(i, i, v) for i, v in enumerate(sample[: self._num_qubits])]
            cost_terms.extend(self._cost_terms)

            circuit = self.build_qaoa_circuit(gamma, beta, cost_terms)
            circuit.measure_all()

            result = self._executor.run(circuit)
            if isinstance(result, list):
                result = result[0]

            cost = self._compute_cost(result.counts, cost_terms)
            scores.append(abs(cost))

        return np.array(scores)

    def _compute_cost(
        self,
        counts: dict[str, int],
        cost_terms: list[tuple[int, int, float]],
    ) -> float:
        """Compute cost from measurement outcomes."""
        total = sum(counts.values())
        avg_cost = 0.0

        for bitstring, count in counts.items():
            bits = [int(b) for b in bitstring[::-1]]
            cost = 0.0

            for i, j, weight in cost_terms:
                if i < len(bits) and j < len(bits):
                    if i == j:
                        cost += weight * (2 * bits[i] - 1)
                    else:
                        cost += weight * (2 * bits[i] - 1) * (2 * bits[j] - 1)

            avg_cost += cost * count / total

        return avg_cost
