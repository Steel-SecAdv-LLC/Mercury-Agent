# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the restored quantum anomaly detector serves.

The package was deleted (740da10a) for defects these tests pin; each fails on
the restored-but-unfixed code. The VQE fit discarded its training data
(held-out AUROC 0.51-0.58 on the data below), at most four features reached
any circuit (an anomaly in feature 6 alone scored at chance, 0.52 on the
kernel path), every scored batch was rescaled to itself (a batch of
anomalies and a batch of normals were both flagged 60 of 60), and the IBM
hardware path was not refused under MERCURY_OFFLINE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.quantum_computing.detector import QuantumAnomalyDetector
from omni_mercury_engine.quantum_computing.executor import QuantumExecutor

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
    assert _auc(scores[:60], scores[60:]) > 0.9  # 0.942 measured


def test_a_feature_beyond_the_qubit_count_reaches_the_circuit(
    vqe: QuantumAnomalyDetector,
) -> None:
    _, normal, anomalous = _data(slice(5, 6))  # feature 6 only, on 3 qubits
    scores = vqe.detect(np.vstack([normal, anomalous])).anomaly_scores
    assert _auc(scores[:60], scores[60:]) > 0.85  # 0.894 measured


def test_batches_are_scaled_by_the_training_data(vqe: QuantumAnomalyDetector) -> None:
    _, normal, anomalous = _data(slice(3, 6))
    flagged_anomalies = int(vqe.detect(anomalous).predictions.sum())
    flagged_normals = int(vqe.detect(normal).predictions.sum())
    assert flagged_anomalies >= 40  # 49 of 60 measured
    assert flagged_normals <= 6  # 3 of 60 measured; contamination 0.05


def test_kernel_path_projects_rather_than_truncates() -> None:
    train, normal, anomalous = _data(slice(5, 6), n_train=30, n_test=15)
    detector = QuantumAnomalyDetector(shots=256, error_mitigation=None)
    detector.fit(train, method="quantum_kernel", num_qubits=2)
    scores = detector.detect(np.vstack([normal, anomalous])).anomaly_scores
    assert _auc(scores[:15], scores[15:]) > 0.9  # 1.0 measured


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
