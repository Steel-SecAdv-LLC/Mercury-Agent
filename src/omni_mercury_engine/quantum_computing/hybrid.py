"""
Hybrid Quantum-Classical Optimization for Mercury Agent.

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

    optimal_parameters: np.ndarray
    optimal_value: float
    n_iterations: int
    convergence_history: list[float]
    final_circuit: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class ClassicalOptimizer:
    """
    Classical optimizer for variational parameter updates.

    Implements gradient-free and gradient-based optimization methods.
    """

    def __init__(
        self,
        method: str = "cobyla",
        maxiter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        """Initialize the optimizer."""
        self._method = method.lower()
        self._maxiter = maxiter
        self._tol = tol

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> tuple[np.ndarray, float, list[float]]:
        """
        Minimize the objective function.

        Args:
            objective: Function to minimize
            initial_params: Initial parameter values

        Returns:
            Tuple of (optimal_params, optimal_value, history)
        """
        if self._method == "cobyla":
            return self._cobyla_minimize(objective, initial_params)
        elif self._method == "spsa":
            return self._spsa_minimize(objective, initial_params)
        elif self._method == "gradient_descent":
            return self._gradient_descent(objective, initial_params)
        else:
            return self._cobyla_minimize(objective, initial_params)

    def _cobyla_minimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> tuple[np.ndarray, float, list[float]]:
        """COBYLA optimization (gradient-free)."""
        params = initial_params.copy()
        history = []
        rho = 1.0

        for iteration in range(self._maxiter):
            current_value = objective(params)
            history.append(current_value)

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

            rho *= 0.95

        return params, history[-1] if history else objective(params), history

    def _spsa_minimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> tuple[np.ndarray, float, list[float]]:
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

            delta = np.random.choice([-1, 1], size=len(params))

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
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> tuple[np.ndarray, float, list[float]]:
        """Gradient descent with finite differences."""
        params = initial_params.copy()
        history = []
        learning_rate = 0.1
        epsilon = 0.01

        for iteration in range(self._maxiter):
            current_value = objective(params)
            history.append(current_value)

            gradient = np.zeros_like(params)
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += epsilon
                gradient[i] = (objective(params_plus) - current_value) / epsilon

            params = params - learning_rate * gradient
            learning_rate *= 0.99

            if iteration > 0 and abs(history[-1] - history[-2]) < self._tol:
                break

        return params, history[-1] if history else objective(params), history


class HybridOptimizer:
    """
    Hybrid quantum-classical optimizer.

    Combines variational quantum circuits with classical optimization.
    """

    def __init__(
        self,
        executor: QuantumExecutor | None = None,
        optimizer_method: str = "cobyla",
        maxiter: int = 100,
    ) -> None:
        """Initialize the hybrid optimizer."""
        if executor is None:
            executor = QuantumExecutor()
        self._executor = executor
        self._optimizer = ClassicalOptimizer(optimizer_method, maxiter)
        self._builder = QuantumCircuitBuilder()

    def optimize(
        self,
        variational_circuit: VariationalCircuit,
        cost_function: Callable[[dict[str, int]], float],
        initial_params: np.ndarray | None = None,
    ) -> OptimizationResult:
        """
        Optimize variational circuit parameters.

        Args:
            variational_circuit: Variational circuit to optimize
            cost_function: Function mapping measurement counts to cost
            initial_params: Initial parameters (random if None)

        Returns:
            OptimizationResult with optimal parameters
        """
        if initial_params is None:
            initial_params = np.random.uniform(0, 2 * np.pi, variational_circuit.num_parameters)

        def objective(params: np.ndarray) -> float:
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
    """
    Quantum kernel for kernel-based machine learning.

    Computes kernel matrix using quantum feature maps.
    """

    def __init__(
        self,
        num_qubits: int,
        feature_map_reps: int = 2,
        executor: QuantumExecutor | None = None,
    ) -> None:
        """Initialize the quantum kernel."""
        self._feature_map = QuantumFeatureMap(num_qubits, feature_map_reps)
        self._executor = executor or QuantumExecutor()
        self._num_qubits = num_qubits

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Compute quantum kernel matrix.

        Args:
            X: First data matrix (n_samples, n_features)
            Y: Second data matrix (default: X)
            shots: Number of shots for kernel estimation

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
        X_train: np.ndarray,
        y_train: np.ndarray,
        C: float = 1.0,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Fit quantum kernel SVM.

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

        def predict(X_test: np.ndarray) -> np.ndarray:
            K_test = self.compute_kernel_matrix(X_test, self._sv_X)
            predictions = np.sign(np.sum(self._sv_alpha * self._sv_y * K_test, axis=1))
            return predictions

        return predict


class VQEAnomalyDetector:
    """
    Variational Quantum Eigensolver for anomaly detection.

    Uses VQE to find ground state of anomaly Hamiltonian.
    """

    def __init__(
        self,
        num_qubits: int,
        ansatz: VariationalAnsatz = VariationalAnsatz.REAL_AMPLITUDES,
        reps: int = 2,
        executor: QuantumExecutor | None = None,
    ) -> None:
        """Initialize the VQE detector."""
        self._num_qubits = num_qubits
        self._variational = VariationalCircuit(num_qubits, ansatz, reps)
        self._optimizer = HybridOptimizer(executor)
        self._encoding = AnomalyEncodingCircuit(num_qubits, EncodingType.ANGLE)
        self._optimal_params: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        maxiter: int = 50,
    ) -> VQEAnomalyDetector:
        """
        Train VQE anomaly detector on normal data.

        Args:
            X_train: Training data (normal samples)
            maxiter: Maximum optimization iterations

        Returns:
            self for method chaining
        """
        np.mean(X_train, axis=0)

        def cost_function(counts: dict[str, int]) -> float:
            total = sum(counts.values())
            energy = 0.0

            for bitstring, count in counts.items():
                parity = sum(int(b) for b in bitstring) % 2
                energy += parity * count / total

            return energy

        result = self._optimizer.optimize(
            self._variational,
            cost_function,
        )

        self._optimal_params = result.optimal_parameters
        self._training_energy = result.optimal_value

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.

        Args:
            X: Data to score

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if self._optimal_params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        scores = []

        for sample in X:
            encoding_circuit = self._encoding.encode(sample[: self._num_qubits])
            variational_circuit = self._variational.build(self._optimal_params)

            builder = QuantumCircuitBuilder()
            combined = builder.create_circuit(self._num_qubits)

            if isinstance(encoding_circuit, SimulatedQuantumCircuit):
                for gate, qubits, params in encoding_circuit._gates:
                    self._apply_gate(combined, gate, qubits, params)
                for gate, qubits, params in variational_circuit._gates:
                    self._apply_gate(combined, gate, qubits, params)
            else:
                combined = encoding_circuit.compose(variational_circuit)

            combined.measure_all()
            result = self._optimizer._executor.run(combined)

            if isinstance(result, list):
                result = result[0]

            energy = self._compute_energy(result.counts)
            scores.append(abs(energy - self._training_energy))

        return np.array(scores)

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
    """
    Quantum Approximate Optimization Algorithm for anomaly detection.

    Uses QAOA to solve combinatorial anomaly problems.
    """

    def __init__(
        self,
        num_qubits: int,
        p: int = 2,
        executor: QuantumExecutor | None = None,
    ) -> None:
        """Initialize the QAOA detector."""
        self._num_qubits = num_qubits
        self._p = p
        self._executor = executor or QuantumExecutor()
        self._optimal_params: np.ndarray | None = None
        self._builder = QuantumCircuitBuilder()

    def build_qaoa_circuit(
        self,
        gamma: list[float],
        beta: list[float],
        cost_terms: list[tuple[int, int, float]],
    ) -> Any:
        """
        Build QAOA circuit.

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
        adjacency_matrix: np.ndarray,
        maxiter: int = 50,
    ) -> QAOAAnomalyDetector:
        """
        Train QAOA on graph-based anomaly problem.

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

        initial_params = np.random.uniform(0, np.pi, 2 * self._p)

        def objective(params: np.ndarray) -> float:
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

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using QAOA.

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
