# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the restored quantum anomaly detector serves.

The package was deleted (740da10a) for defects these tests pin; each fails on
the restored-but-unfixed code. The VQE fit discarded its training data
(held-out AUROC 0.51-0.58 on the data below), at most four features reached
any circuit (an anomaly in feature 6 alone scored AUROC 0.56-0.68 on the
kernel path), every scored batch was rescaled to itself (a batch of
anomalies and a batch of normals were both flagged 60 of 60), and the IBM
hardware path was not refused under MERCURY_OFFLINE.

The training-fitted scaling then put out-of-range samples outside [0, 1],
where the pi * x encodings repeat with period 2: a sample two training ranges
out encoded exactly like one in range. Out-of-range values are squashed into
(-0.45, 1.45), and the figures below were measured with that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.quantum_computing.detector import QuantumAnomalyDetector
from omni_mercury_engine.quantum_computing.executor import QuantumExecutor
from omni_mercury_engine.quantum_computing.hybrid import QuantumKernel

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _auc(normal: NDArray[np.float64], anomalous: NDArray[np.float64]) -> float:
    """Mann-Whitney ROC AUC, ties counting one half."""
    greater = (anomalous[:, None] > normal[None, :]).sum()
    ties = (anomalous[:, None] == normal[None, :]).sum()
    return float((greater + 0.5 * ties) / (anomalous.size * normal.size))


def _data(
    shift_features: slice, n_train: int = 120, n_test: int = 60
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    train = rng.normal(0.3, 0.05, (n_train, 6))
    normal = rng.normal(0.3, 0.05, (n_test, 6))
    anomalous = rng.normal(0.3, 0.05, (n_test, 6))
    anomalous[:, shift_features] += 0.35
    return train, normal, anomalous


@pytest.fixture(scope="module")
def vqe() -> QuantumAnomalyDetector:
    train, _, _ = _data(slice(3, 6))
    detector = QuantumAnomalyDetector(error_mitigation=None)
    return detector.fit(train, method="vqe_anomaly", num_qubits=3, maxiter=60)


def test_vqe_learns_the_training_data(vqe: QuantumAnomalyDetector) -> None:
    _, normal, anomalous = _data(slice(3, 6))
    scores = vqe.detect(np.vstack([normal, anomalous])).anomaly_scores
    assert _auc(scores[:60], scores[60:]) > 0.9  # 0.997 measured (0.942 before the squash)


def test_a_feature_beyond_the_qubit_count_reaches_the_circuit(
    vqe: QuantumAnomalyDetector,
) -> None:
    _, normal, anomalous = _data(slice(5, 6))  # feature 6 only, on 3 qubits
    scores = vqe.detect(np.vstack([normal, anomalous])).anomaly_scores
    assert _auc(scores[:60], scores[60:]) > 0.85  # 0.998 measured


def test_batches_are_scaled_by_the_training_data(vqe: QuantumAnomalyDetector) -> None:
    _, normal, anomalous = _data(slice(3, 6))
    flagged_anomalies = int(vqe.detect(anomalous).predictions.sum())
    flagged_normals = int(vqe.detect(normal).predictions.sum())
    assert flagged_anomalies >= 40  # 59 of 60 measured
    assert flagged_normals <= 6  # 3 of 60 measured; contamination 0.05


def _kernel_scores(seed: int) -> NDArray[np.float64]:
    train, normal, anomalous = _data(slice(5, 6), n_train=30, n_test=15)
    detector = QuantumAnomalyDetector(shots=256, error_mitigation=None, seed=seed)
    detector.fit(train, method="quantum_kernel", num_qubits=2)
    return np.asarray(detector.detect(np.vstack([normal, anomalous])).anomaly_scores)


def test_kernel_path_projects_rather_than_truncates() -> None:
    scores = _kernel_scores(seed=0)
    # 0.922 at seed 0, 0.862-0.938 over seeds 0-19 (truncation: 0.57-0.66 over 20 runs).
    assert _auc(scores[:15], scores[15:]) > 0.8


def test_seeded_kernel_is_reproducible() -> None:
    """Fails on the unseeded shot sampler, whose scores changed run to run."""
    assert np.array_equal(_kernel_scores(seed=3), _kernel_scores(seed=3))


def test_exact_kernel_is_the_fidelity() -> None:
    """``shots=None`` gives |<phi(x1)|phi(x2)>|^2 exactly; shots estimate it."""
    kernel = QuantumKernel(2, seed=1)
    points = np.random.default_rng(4).uniform(0.0, 1.0, (5, 2))
    exact = kernel.compute_kernel_matrix(points, shots=None)
    assert np.allclose(np.diag(exact), 1.0)
    assert np.allclose(exact, exact.T)
    assert np.all((exact >= 0.0) & (exact <= 1.0 + 1e-12))
    shots = 4096
    sampled = kernel.compute_kernel_matrix(points, shots=shots)
    standard_error = np.sqrt(exact * (1.0 - exact) / shots) + 1e-12
    assert np.all(np.abs(sampled - exact) <= 5.0 * standard_error + 1.0 / shots)


@pytest.mark.parametrize("ranges_out", [2.0, 4.0])
def test_far_out_of_range_samples_are_not_aliased(
    vqe: QuantumAnomalyDetector, ranges_out: float
) -> None:
    """Fails without the squash: at two training ranges out, 3 of 60 were flagged."""
    train, normal, _ = _data(slice(3, 6))
    far = normal.copy()
    far[:, 3:6] += ranges_out * (train.max(axis=0) - train.min(axis=0))[3:6]
    assert int(vqe.detect(far).predictions.sum()) >= 55  # 60 of 60 measured


def test_refit_does_not_inherit_the_vqe_threshold() -> None:
    train, _, _ = _data(slice(3, 6), n_train=30)
    detector = QuantumAnomalyDetector(shots=64, error_mitigation=None)
    detector.fit(train, method="vqe_anomaly", num_qubits=2, maxiter=5)
    detector.fit(train, method="quantum_kernel", num_qubits=2)
    assert detector._threshold == 0.5


def test_detect_refuses_non_finite_data(vqe: QuantumAnomalyDetector) -> None:
    sample = np.full((1, 6), 0.3)
    sample[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        vqe.detect(sample)


def test_vqe_on_an_asynchronous_backend_says_so() -> None:
    from omni_mercury_engine.quantum_computing.hybrid import VQEAnomalyDetector

    class Asynchronous:
        is_simulator = False

        def run(self, _circuits: object, shots: int | None = None) -> list[object]:
            return []

    detector = VQEAnomalyDetector(2, executor=Asynchronous())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="asynchronously"):
        detector.fit(np.random.default_rng(0).random((5, 2)), maxiter=3)


@pytest.mark.parametrize("method", ["coordinate_search", "gradient_descent"])
def test_optimizers_report_the_value_at_the_returned_point(method: str) -> None:
    from omni_mercury_engine.quantum_computing.hybrid import ClassicalOptimizer

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    params, value, history = ClassicalOptimizer(method, maxiter=3, seed=0).minimize(
        objective, np.array([3.0, -2.0])
    )
    assert value == pytest.approx(objective(params))
    assert history[-1] == value


def test_ibm_hardware_is_refused_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    from omni_mercury_engine.datasets.exceptions import OfflineModeError
    from omni_mercury_engine.quantum_computing.circuits import SimulatedQuantumCircuit

    monkeypatch.setenv("MERCURY_OFFLINE", "1")
    executor = QuantumExecutor(
        backend="ibm_brisbane", api_token="test-token"  # noqa: S106 - sanitized test value
    )
    assert executor.is_simulator is False
    circuit = SimulatedQuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()
    with pytest.raises(OfflineModeError, match=r"quantum\.ibm\.com"):
        executor.run(circuit)


def test_simulator_runs_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    from omni_mercury_engine.quantum_computing.circuits import SimulatedQuantumCircuit

    monkeypatch.setenv("MERCURY_OFFLINE", "1")
    executor = QuantumExecutor(backend="aer_simulator")
    assert executor.is_simulator is True
    circuit = SimulatedQuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    result = executor.run(circuit)
    if isinstance(result, list):
        result = result[0]
    assert set(result.counts) == {"1"}
