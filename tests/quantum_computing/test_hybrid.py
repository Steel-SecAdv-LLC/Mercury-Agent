# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic unit tests for the hybrid quantum-classical optimizers.

Exercises the public surface of
:mod:`omni_mercury_engine.quantum_computing.hybrid`:

* :class:`OptimizationResult` -- the result dataclass and its ``metadata``
  default.
* :class:`ClassicalOptimizer` -- construction and every ``minimize`` method
  (COBYLA, SPSA, gradient descent, the unknown-method COBYLA fallback), the
  convergence-break and empty-history (``maxiter == 0``) edge paths, and
  seed-driven reproducibility of the stochastic SPSA branch.
* :class:`HybridOptimizer` -- construction wiring, ``optimize`` with and
  without caller-supplied ``initial_params``, and the list-result unwrap
  branch driven by a deterministic stub executor.
* :class:`QuantumKernel` -- kernel-matrix computation (self and cross), the
  fidelity self-kernel invariant (``K[i, i] == 1``), the [0, 1] range
  contract, and ``fit_svm`` + prediction.
* :class:`VQEAnomalyDetector` / :class:`QAOAAnomalyDetector` -- construction,
  ``fit`` chaining, ``score`` shape/finiteness invariants, the
  not-fitted ``ValueError`` guard, the private gate-application and
  energy/cost helpers asserted to exact values, and the list-result unwrap
  branches.

The module ships a pure-NumPy simulation fallback (Qiskit is not installed),
so no network, hardware, or Qiskit dependency is touched.  The NumPy
statevector sampler seeds itself from OS entropy, so the end-to-end
fit/score tests assert only the *sampling-invariant* contract (shape,
dtype, sign, finiteness) rather than exact scores; the pure-NumPy optimizer
runs over analytic objectives, the closed-form energy/cost helpers, and the
fidelity self-kernel are fully deterministic and are asserted to exact
values.  All input data is drawn from a seeded ``numpy.random.Generator`` so
every deterministic assertion is reproducible and free of wall-clock or
network dependence.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from omni_mercury_engine.quantum_computing.circuits import (
    QuantumCircuitBuilder,
    QuantumFeatureMap,
    SimulatedQuantumCircuit,
    VariationalAnsatz,
    VariationalCircuit,
)
from omni_mercury_engine.quantum_computing.executor import QuantumExecutor
from omni_mercury_engine.quantum_computing.hybrid import (
    ClassicalOptimizer,
    HybridOptimizer,
    OptimizationResult,
    QAOAAnomalyDetector,
    QuantumKernel,
    VQEAnomalyDetector,
)

SEED = 20240521

# Small, cheap defaults keep the NumPy statevector simulation trivially fast
# while still exercising every code path.
FAST_QUBITS = 2
FAST_SHOTS = 32
FEW_ITERS = 3


def _rng() -> np.random.Generator:
    """Return a freshly seeded generator for reproducible input data."""
    return np.random.default_rng(SEED)


def _sample_matrix(n_samples: int = 4, n_features: int = 2) -> np.ndarray:
    """Reproducible feature matrix already inside the [0, 1) range."""
    return _rng().random((n_samples, n_features))


def _parity_cost(counts: dict[str, int]) -> float:
    """Mean bit-parity of the measured bitstrings -- a valid cost function."""
    total = sum(counts.values())
    return sum((sum(int(b) for b in bs) % 2) * c / total for bs, c in counts.items())


class _ListResultExecutor:
    """Executor whose ``run`` returns a *list* of one fixed result.

    Drives the ``isinstance(result, list)`` unwrap branch that the real
    simulator (single-circuit -> single result) never reaches, while keeping
    the returned counts deterministic.  Only ``.counts`` is ever read.
    """

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = dict(counts)

    def run(self, circuit: Any) -> list[Any]:
        return [SimpleNamespace(counts=dict(self._counts))]


# --------------------------------------------------------------------------- #
# OptimizationResult
# --------------------------------------------------------------------------- #
class TestOptimizationResult:
    """The result dataclass and its ``metadata`` default factory."""

    def test_metadata_defaults_to_empty_dict(self) -> None:
        result = OptimizationResult(
            optimal_parameters=np.array([1.0, 2.0]),
            optimal_value=0.5,
            n_iterations=3,
            convergence_history=[1.0, 0.5],
            final_circuit=object(),
        )
        assert result.metadata == {}
        assert result.n_iterations == 3
        assert result.optimal_value == pytest.approx(0.5)
        np.testing.assert_array_equal(result.optimal_parameters, [1.0, 2.0])
        assert result.convergence_history == [1.0, 0.5]

    def test_metadata_default_not_shared(self) -> None:
        first = OptimizationResult(np.array([0.0]), 0.0, 0, [], None)
        second = OptimizationResult(np.array([0.0]), 0.0, 0, [], None)
        first.metadata["x"] = 1
        assert second.metadata == {}

    def test_explicit_metadata(self) -> None:
        result = OptimizationResult(
            optimal_parameters=np.array([0.0]),
            optimal_value=1.0,
            n_iterations=1,
            convergence_history=[1.0],
            final_circuit=None,
            metadata={"method": "cobyla"},
        )
        assert result.metadata == {"method": "cobyla"}


# --------------------------------------------------------------------------- #
# ClassicalOptimizer
# --------------------------------------------------------------------------- #
class TestClassicalOptimizerConstruction:
    """Constructor stores lowercased method and wires a seeded generator."""

    def test_defaults(self) -> None:
        opt = ClassicalOptimizer()
        assert opt._method == "cobyla"
        assert opt._maxiter == 100
        assert opt._tol == pytest.approx(1e-6)
        assert isinstance(opt._rng, np.random.Generator)

    def test_method_is_lowercased(self) -> None:
        assert ClassicalOptimizer(method="COBYLA")._method == "cobyla"
        assert ClassicalOptimizer(method="SPSA")._method == "spsa"

    def test_custom_maxiter_and_tol(self) -> None:
        opt = ClassicalOptimizer(method="spsa", maxiter=7, tol=1e-3)
        assert opt._maxiter == 7
        assert opt._tol == pytest.approx(1e-3)


def _quadratic(center: float = 1.0) -> Callable[[np.ndarray], float]:
    """A convex analytic objective with its minimum at ``center`` per axis."""

    def obj(params: np.ndarray) -> float:
        return float(np.sum((params - center) ** 2))

    return obj


class TestClassicalOptimizerMinimize:
    """Every ``minimize`` branch over pure-NumPy analytic objectives.

    These objectives touch no quantum sampler, so results are fully
    deterministic under the instance seed.
    """

    def test_cobyla_reduces_objective_and_returns_shapes(self) -> None:
        opt = ClassicalOptimizer("cobyla", maxiter=20, seed=SEED)
        start = np.array([0.0, 0.0])
        params, value, history = opt.minimize(_quadratic(1.0), start)
        assert isinstance(params, np.ndarray)
        assert params.shape == start.shape
        assert isinstance(history, list)
        assert value == pytest.approx(history[-1])
        # COBYLA must not leave the objective worse than the starting point.
        assert value <= history[0]

    def test_cobyla_negative_direction_branch(self) -> None:
        # Minimum below the start forces the ``-rho`` trial branch (the ``+rho``
        # step makes things worse, so the optimizer steps the other way).
        opt = ClassicalOptimizer("cobyla", maxiter=15, seed=SEED)
        params, value, history = opt.minimize(_quadratic(-5.0), np.array([0.0, 0.0]))
        assert value < history[0]
        assert np.all(params < 0.0)

    def test_gradient_descent_reduces_objective(self) -> None:
        opt = ClassicalOptimizer("gradient_descent", maxiter=40, seed=SEED)
        params, value, history = opt.minimize(_quadratic(1.0), np.array([0.0, 0.0]))
        assert value < history[0]
        # Converges toward the analytic minimum at [1, 1].
        np.testing.assert_allclose(params, [1.0, 1.0], atol=0.2)

    def test_spsa_is_reproducible_under_same_seed(self) -> None:
        obj = _quadratic(1.0)
        start = np.array([0.0, 0.0])
        a = ClassicalOptimizer("spsa", maxiter=12, seed=SEED).minimize(obj, start)
        b = ClassicalOptimizer("spsa", maxiter=12, seed=SEED).minimize(obj, start)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]
        assert a[2] == b[2]

    def test_spsa_differs_across_seeds(self) -> None:
        obj = _quadratic(1.0)
        start = np.array([0.0, 0.0])
        a = ClassicalOptimizer("spsa", maxiter=12, seed=SEED).minimize(obj, start)
        b = ClassicalOptimizer("spsa", maxiter=12, seed=SEED + 1).minimize(obj, start)
        assert not np.array_equal(a[0], b[0])

    def test_unknown_method_falls_back_to_cobyla(self) -> None:
        # An unrecognized method routes through the COBYLA branch, so it behaves
        # identically to an explicit "cobyla" run under the same seed.
        obj = _quadratic(1.0)
        start = np.array([0.0, 0.0])
        unknown = ClassicalOptimizer("mystery", maxiter=10, seed=SEED).minimize(obj, start)
        cobyla = ClassicalOptimizer("cobyla", maxiter=10, seed=SEED).minimize(obj, start)
        np.testing.assert_array_equal(unknown[0], cobyla[0])
        assert unknown[2] == cobyla[2]

    @pytest.mark.parametrize("method", ["cobyla", "spsa", "gradient_descent"])
    def test_constant_objective_triggers_convergence_break(self, method: str) -> None:
        # A flat objective makes |history[-1] - history[-2]| == 0 < tol, so the
        # loop breaks on the second iteration for every method.
        opt = ClassicalOptimizer(method, maxiter=50, tol=1e-6, seed=SEED)
        _, value, history = opt.minimize(lambda p: 7.0, np.array([1.0]))
        assert len(history) == 2
        assert value == pytest.approx(7.0)

    @pytest.mark.parametrize("method", ["cobyla", "spsa", "gradient_descent"])
    def test_maxiter_zero_uses_objective_fallback(self, method: str) -> None:
        # With maxiter == 0 the loop never runs, so history stays empty and the
        # returned value comes from the ``objective(params)`` fallback.
        opt = ClassicalOptimizer(method, maxiter=0, seed=SEED)
        params, value, history = opt.minimize(_quadratic(0.0), np.array([2.0, 3.0]))
        assert history == []
        assert value == pytest.approx(13.0)  # 2^2 + 3^2
        np.testing.assert_array_equal(params, [2.0, 3.0])

    @pytest.mark.parametrize("method", ["cobyla", "spsa", "gradient_descent"])
    def test_empty_parameter_vector(self, method: str) -> None:
        # Zero-length parameters exercise the per-coordinate loops with no body.
        opt = ClassicalOptimizer(method, maxiter=3, seed=SEED)
        params, value, history = opt.minimize(lambda p: 1.0, np.array([]))
        assert params.shape == (0,)
        assert value == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# HybridOptimizer
# --------------------------------------------------------------------------- #
def _variational(num_qubits: int = FAST_QUBITS, reps: int = 1) -> VariationalCircuit:
    return VariationalCircuit(num_qubits, VariationalAnsatz.REAL_AMPLITUDES, reps)


class TestHybridOptimizerConstruction:
    """Constructor wiring and default component types."""

    def test_default_construction(self) -> None:
        opt = HybridOptimizer()
        assert isinstance(opt._executor, QuantumExecutor)
        assert isinstance(opt._optimizer, ClassicalOptimizer)
        assert isinstance(opt._builder, QuantumCircuitBuilder)
        assert isinstance(opt._rng, np.random.Generator)

    def test_optimizer_method_propagates(self) -> None:
        opt = HybridOptimizer(optimizer_method="SPSA", maxiter=9)
        assert opt._optimizer._method == "spsa"
        assert opt._optimizer._maxiter == 9

    def test_supplied_executor_is_used(self) -> None:
        stub = _ListResultExecutor({"00": 1})
        opt = HybridOptimizer(executor=stub)  # type: ignore[arg-type]
        assert opt._executor is stub  # type: ignore[comparison-overlap]


class TestHybridOptimizerOptimize:
    """End-to-end ``optimize`` over the real simulator and stub executor."""

    def test_returns_optimization_result_with_expected_shapes(self) -> None:
        circuit = _variational()
        opt = HybridOptimizer(optimizer_method="cobyla", maxiter=FEW_ITERS, seed=SEED)
        result = opt.optimize(circuit, _parity_cost)
        assert isinstance(result, OptimizationResult)
        assert result.optimal_parameters.shape == (circuit.num_parameters,)
        assert isinstance(result.optimal_value, float)
        assert result.n_iterations == len(result.convergence_history)
        assert result.n_iterations == FEW_ITERS
        assert isinstance(result.final_circuit, SimulatedQuantumCircuit)
        assert np.isfinite(result.optimal_value)

    def test_random_initial_params_have_circuit_length(self) -> None:
        circuit = _variational(num_qubits=3, reps=2)
        opt = HybridOptimizer(maxiter=2, seed=SEED)
        result = opt.optimize(circuit, _parity_cost)
        assert result.optimal_parameters.shape == (circuit.num_parameters,)

    def test_supplied_initial_params_are_honored(self) -> None:
        circuit = _variational()
        opt = HybridOptimizer(maxiter=2, seed=SEED)
        init = np.zeros(circuit.num_parameters)
        result = opt.optimize(circuit, _parity_cost, initial_params=init)
        assert result.optimal_parameters.shape == init.shape

    def test_list_result_is_unwrapped(self) -> None:
        # The stub executor returns a list, forcing the ``result = result[0]``
        # unwrap; the fixed all-odd-parity counts make the cost exactly 1.0.
        circuit = _variational()
        opt = HybridOptimizer(
            executor=_ListResultExecutor({"01": 10}),  # type: ignore[arg-type]
            optimizer_method="cobyla",
            maxiter=2,
            seed=SEED,
        )
        result = opt.optimize(circuit, _parity_cost)
        assert result.optimal_value == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# QuantumKernel
# --------------------------------------------------------------------------- #
class TestQuantumKernelConstruction:
    """Constructor wiring and default component types."""

    def test_default_construction(self) -> None:
        kernel = QuantumKernel(FAST_QUBITS)
        assert isinstance(kernel._feature_map, QuantumFeatureMap)
        assert isinstance(kernel._executor, QuantumExecutor)
        assert kernel._num_qubits == FAST_QUBITS

    def test_supplied_executor_is_used(self) -> None:
        stub = _ListResultExecutor({"00": 1})
        kernel = QuantumKernel(FAST_QUBITS, executor=stub)  # type: ignore[arg-type]
        assert kernel._executor is stub  # type: ignore[comparison-overlap]


class TestQuantumKernelMatrix:
    """Kernel-matrix computation and the fidelity self-kernel invariant."""

    def test_gram_matrix_shape_and_unit_diagonal(self) -> None:
        kernel = QuantumKernel(FAST_QUBITS, feature_map_reps=1)
        X = _sample_matrix(n_samples=3, n_features=2)
        K = kernel.compute_kernel_matrix(X, shots=FAST_SHOTS)
        assert K.shape == (3, 3)
        # A state's fidelity with itself is exactly 1 (U U^-1 == identity), so
        # the diagonal is deterministic regardless of shot noise.
        np.testing.assert_allclose(np.diag(K), np.ones(3))

    def test_kernel_values_within_unit_range(self) -> None:
        kernel = QuantumKernel(FAST_QUBITS, feature_map_reps=1)
        X = _sample_matrix(n_samples=3, n_features=2)
        K = kernel.compute_kernel_matrix(X, shots=FAST_SHOTS)
        assert np.all(K >= 0.0)
        assert np.all(K <= 1.0)

    def test_cross_kernel_shape_when_y_supplied(self) -> None:
        kernel = QuantumKernel(FAST_QUBITS, feature_map_reps=1)
        X = _sample_matrix(n_samples=3, n_features=2)
        Y = _sample_matrix(n_samples=2, n_features=2)
        K = kernel.compute_kernel_matrix(X, Y, shots=FAST_SHOTS)
        assert K.shape == (3, 2)
        assert np.all((K >= 0.0) & (K <= 1.0))


class TestQuantumKernelSVM:
    """``fit_svm`` trains a callable predictor over the quantum kernel."""

    def test_fit_returns_callable_and_stores_support_vectors(self) -> None:
        kernel = QuantumKernel(FAST_QUBITS, feature_map_reps=1)
        X = _sample_matrix(n_samples=4, n_features=2)
        y = np.array([1.0, 1.0, -1.0, -1.0])
        predict = kernel.fit_svm(X, y, C=1.0)
        assert callable(predict)
        # Every alpha is clipped into [0, C]; the retained support vectors are
        # the strictly-positive ones, so their alphas are all > 0.
        assert kernel._sv_alpha.ndim == 1
        assert np.all(kernel._sv_alpha > 1e-5)
        assert kernel._sv_X.shape[0] == kernel._sv_alpha.shape[0]

    def test_prediction_shape_and_sign_values(self) -> None:
        kernel = QuantumKernel(FAST_QUBITS, feature_map_reps=1)
        X = _sample_matrix(n_samples=4, n_features=2)
        y = np.array([1.0, 1.0, -1.0, -1.0])
        predict = kernel.fit_svm(X, y)
        preds = predict(_sample_matrix(n_samples=3, n_features=2))
        assert preds.shape == (3,)
        # np.sign yields only {-1, 0, 1}.
        assert set(np.unique(preds)).issubset({-1.0, 0.0, 1.0})


# --------------------------------------------------------------------------- #
# VQEAnomalyDetector
# --------------------------------------------------------------------------- #
class TestVQEConstruction:
    """Constructor wiring and initial un-fitted state."""

    def test_default_construction(self) -> None:
        vqe = VQEAnomalyDetector(FAST_QUBITS)
        assert vqe._num_qubits == FAST_QUBITS
        assert isinstance(vqe._variational, VariationalCircuit)
        assert isinstance(vqe._optimizer, HybridOptimizer)
        assert vqe._optimal_params is None

    def test_custom_ansatz(self) -> None:
        vqe = VQEAnomalyDetector(FAST_QUBITS, ansatz=VariationalAnsatz.TWO_LOCAL, reps=1)
        assert vqe._variational._ansatz is VariationalAnsatz.TWO_LOCAL


class TestVQEFitScore:
    """Training and scoring contract of the VQE detector."""

    def test_fit_returns_self_and_stores_params(self) -> None:
        vqe = VQEAnomalyDetector(FAST_QUBITS, reps=1)
        returned = vqe.fit(_sample_matrix(), maxiter=FEW_ITERS)
        assert returned is vqe
        assert vqe._optimal_params is not None
        assert vqe._optimal_params.shape == (vqe._variational.num_parameters,)
        assert isinstance(vqe._training_energy, float)

    def test_score_shape_and_nonnegative_finite(self) -> None:
        vqe = VQEAnomalyDetector(FAST_QUBITS, reps=1)
        vqe.fit(_sample_matrix(), maxiter=FEW_ITERS)
        scores = vqe.score(_sample_matrix(n_samples=3, n_features=2))
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))
        # Scores are ``abs(energy - training_energy)`` -> non-negative.
        assert np.all(scores >= 0.0)

    def test_score_before_fit_raises(self) -> None:
        vqe = VQEAnomalyDetector(FAST_QUBITS)
        with pytest.raises(ValueError, match="Model not fitted"):
            vqe.score(_sample_matrix(n_samples=2, n_features=2))

    def test_score_list_result_is_unwrapped(self) -> None:
        # Replace the inner executor with a stub returning a list, so the
        # ``result = result[0]`` unwrap in score() is exercised.  Counts of
        # even-parity "11" give energy 0, so each score is |0 - training|.
        vqe = VQEAnomalyDetector(FAST_QUBITS, reps=1)
        vqe.fit(_sample_matrix(), maxiter=FEW_ITERS)
        vqe._optimizer._executor = _ListResultExecutor({"11": 100})  # type: ignore[assignment]
        scores = vqe.score(np.array([[0.1, 0.2], [0.3, 0.4]]))
        expected = abs(0.0 - vqe._training_energy)
        np.testing.assert_allclose(scores, [expected, expected])


class TestVQEHelpers:
    """The private gate-application and energy helpers are pure/deterministic."""

    def test_apply_gate_covers_every_supported_gate(self) -> None:
        vqe = VQEAnomalyDetector(2)
        circuit = SimulatedQuantumCircuit(2)
        vqe._apply_gate(circuit, "h", [0], [])
        vqe._apply_gate(circuit, "x", [1], [])
        vqe._apply_gate(circuit, "rx", [0], [0.3])
        vqe._apply_gate(circuit, "ry", [0], [0.4])
        vqe._apply_gate(circuit, "rz", [1], [0.5])
        vqe._apply_gate(circuit, "cx", [0, 1], [])
        vqe._apply_gate(circuit, "cz", [0, 1], [])
        recorded = [g[0] for g in circuit._gates]
        assert recorded == ["h", "x", "rx", "ry", "rz", "cx", "cz"]

    def test_apply_gate_ignores_unknown_gate(self) -> None:
        vqe = VQEAnomalyDetector(2)
        circuit = SimulatedQuantumCircuit(2)
        vqe._apply_gate(circuit, "unknown_gate", [0], [])
        assert circuit._gates == []

    def test_compute_energy_all_even_parity_is_zero(self) -> None:
        vqe = VQEAnomalyDetector(2)
        # "00" and "11" both have even bit-parity -> energy contribution 0.
        assert vqe._compute_energy({"00": 50, "11": 50}) == pytest.approx(0.0)

    def test_compute_energy_all_odd_parity_is_one(self) -> None:
        vqe = VQEAnomalyDetector(2)
        assert vqe._compute_energy({"01": 100}) == pytest.approx(1.0)

    def test_compute_energy_mixed_parity_is_fraction(self) -> None:
        vqe = VQEAnomalyDetector(2)
        # 1 odd out of 4 shots -> 0.25.
        assert vqe._compute_energy({"00": 3, "01": 1}) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# QAOAAnomalyDetector
# --------------------------------------------------------------------------- #
class TestQAOAConstruction:
    """Constructor wiring and initial un-fitted state."""

    def test_default_construction(self) -> None:
        qaoa = QAOAAnomalyDetector(FAST_QUBITS)
        assert qaoa._num_qubits == FAST_QUBITS
        assert qaoa._p == 2
        assert isinstance(qaoa._executor, QuantumExecutor)
        assert isinstance(qaoa._builder, QuantumCircuitBuilder)
        assert isinstance(qaoa._rng, np.random.Generator)
        assert qaoa._optimal_params is None

    def test_supplied_executor_is_used(self) -> None:
        stub = _ListResultExecutor({"00": 1})
        qaoa = QAOAAnomalyDetector(FAST_QUBITS, executor=stub)  # type: ignore[arg-type]
        assert qaoa._executor is stub  # type: ignore[comparison-overlap]


class TestQAOACircuit:
    """The QAOA circuit builder assembles the expected structure."""

    def test_build_qaoa_circuit_gate_layout(self) -> None:
        qaoa = QAOAAnomalyDetector(2, p=1)
        circuit = qaoa.build_qaoa_circuit([0.5], [0.3], [(0, 1, 1.0)])
        assert isinstance(circuit, SimulatedQuantumCircuit)
        assert circuit.num_qubits == 2
        gate_names = [g[0] for g in circuit._gates]
        # 2 initial H, then one cost layer (cx, rz, cx) and 2 mixer RX gates.
        assert gate_names == ["h", "h", "cx", "rz", "cx", "rx", "rx"]

    def test_build_qaoa_circuit_multiple_layers(self) -> None:
        qaoa = QAOAAnomalyDetector(2, p=2)
        circuit = qaoa.build_qaoa_circuit([0.1, 0.2], [0.3, 0.4], [(0, 1, 1.0)])
        gate_names = [g[0] for g in circuit._gates]
        assert gate_names.count("rx") == 4  # 2 qubits * p=2 mixer layers
        assert gate_names.count("cx") == 4  # 2 per cost term * p=2 layers


class TestQAOAFitScore:
    """Training and scoring contract of the QAOA detector."""

    def test_fit_returns_self_and_builds_cost_terms(self) -> None:
        qaoa = QAOAAnomalyDetector(3, p=1, seed=SEED)
        adjacency = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )
        returned = qaoa.fit(adjacency, maxiter=FEW_ITERS)
        assert returned is qaoa
        assert qaoa._optimal_params is not None
        assert qaoa._optimal_params.shape == (2 * qaoa._p,)
        # Only the non-zero upper-triangular edges become cost terms.
        edges = {(i, j) for i, j, _ in qaoa._cost_terms}
        assert edges == {(0, 1), (1, 2)}

    def test_fit_with_zero_adjacency_yields_no_cost_terms(self) -> None:
        qaoa = QAOAAnomalyDetector(2, p=1, seed=SEED)
        qaoa.fit(np.zeros((2, 2)), maxiter=2)
        assert qaoa._cost_terms == []
        assert qaoa._optimal_params is not None
        assert qaoa._optimal_params.shape == (2,)

    def test_score_shape_and_nonnegative_finite(self) -> None:
        qaoa = QAOAAnomalyDetector(3, p=1, seed=SEED)
        adjacency = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )
        qaoa.fit(adjacency, maxiter=FEW_ITERS)
        scores = qaoa.score(_sample_matrix(n_samples=3, n_features=3))
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))
        # Scores are ``abs(cost)`` -> non-negative.
        assert np.all(scores >= 0.0)

    def test_score_before_fit_raises(self) -> None:
        qaoa = QAOAAnomalyDetector(FAST_QUBITS)
        with pytest.raises(ValueError, match="Model not fitted"):
            qaoa.score(_sample_matrix(n_samples=2, n_features=2))

    def test_fit_and_score_unwrap_list_result(self) -> None:
        # A stub executor returning a list drives the ``result = result[0]``
        # unwrap in both the fit objective and score().
        qaoa = QAOAAnomalyDetector(
            FAST_QUBITS,
            p=1,
            executor=_ListResultExecutor({"00": 100}),  # type: ignore[arg-type]
            seed=SEED,
        )
        qaoa.fit(np.array([[0.0, 1.0], [1.0, 0.0]]), maxiter=2)
        scores = qaoa.score(np.array([[0.5, 0.5]]))
        assert scores.shape == (1,)
        assert np.all(np.isfinite(scores))


class TestQAOACostHelper:
    """``_compute_cost`` is a pure, closed-form function of the counts."""

    def test_pairwise_zz_correlation(self) -> None:
        qaoa = QAOAAnomalyDetector(2, p=1)
        # For ZZ term (0, 1): "00" and "11" are aligned (+1), so cost == weight.
        cost = qaoa._compute_cost({"00": 50, "11": 50}, [(0, 1, 1.0)])
        assert cost == pytest.approx(1.0)

    def test_anti_aligned_pair_flips_sign(self) -> None:
        qaoa = QAOAAnomalyDetector(2, p=1)
        # "01" is anti-aligned -> (+1)*(-1) == -1 times the weight.
        cost = qaoa._compute_cost({"01": 100}, [(0, 1, 1.0)])
        assert cost == pytest.approx(-1.0)

    def test_diagonal_single_qubit_term(self) -> None:
        qaoa = QAOAAnomalyDetector(2, p=1)
        # i == j linear term: weight * (2*bit - 1); bit 1 -> +weight.
        cost = qaoa._compute_cost({"11": 100}, [(0, 0, 2.0)])
        assert cost == pytest.approx(2.0)

    def test_out_of_range_indices_are_skipped(self) -> None:
        qaoa = QAOAAnomalyDetector(2, p=1)
        # Index 5 exceeds the 2-bit strings, so the term is skipped and the
        # in-range term alone determines the cost.
        cost = qaoa._compute_cost({"11": 100}, [(0, 5, 1.0), (0, 1, 1.0)])
        assert cost == pytest.approx(1.0)

    def test_mixed_outcomes_average_over_shots(self) -> None:
        qaoa = QAOAAnomalyDetector(2, p=1)
        # Half aligned (+1), half anti-aligned (-1) -> average 0.
        cost = qaoa._compute_cost({"00": 50, "01": 50}, [(0, 1, 1.0)])
        assert cost == pytest.approx(0.0)
