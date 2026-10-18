# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic unit tests for the quantum anomaly-detection facade.

Exercises the public surface of
:mod:`omni_mercury_engine.quantum_computing.detector`:

* :class:`QuantumAnomalyDetector` -- construction, ``fit`` for every
  supported method, ``detect`` across the quantum pipelines plus the
  classical-fallback / error paths, and ``estimate_resources``.
* the :class:`QuantumDetectionResult` and :class:`QuantumResourceEstimate`
  result dataclasses (field defaults and explicit construction).

The module ships a pure-NumPy simulation fallback (Qiskit is not
installed), so no network, hardware, or Qiskit dependency is touched.
The NumPy statevector sampler seeds itself from OS entropy, so the
quantum-pipeline tests assert only the *sampling-invariant* contract
(shapes, dtypes, sign, finiteness, metadata) rather than exact scores;
the classical z-score fallback and the closed-form ``estimate_resources``
arithmetic are fully deterministic and are asserted to exact values.

All input data is drawn from a seeded ``numpy.random.Generator`` so the
inputs -- and therefore every deterministic assertion -- are reproducible
and free of wall-clock or network dependence.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from omni_mercury_engine.quantum_computing.circuits import ErrorMitigationCircuit
from omni_mercury_engine.quantum_computing.detector import (
    QuantumAnomalyDetector,
    QuantumDetectionResult,
    QuantumResourceEstimate,
)
from omni_mercury_engine.quantum_computing.executor import QuantumExecutor
from omni_mercury_engine.quantum_computing.hybrid import (
    QAOAAnomalyDetector,
    VQEAnomalyDetector,
)

SEED = 20240521

# A small, cheap default: 2 qubits + few shots keeps the NumPy statevector
# simulation trivially fast while still exercising every code path.
FAST_SHOTS = 16
FAST_QUBITS = 2


def _rng() -> np.random.Generator:
    """Return a freshly seeded generator for reproducible input data."""
    return np.random.default_rng(SEED)


def _sample_matrix(n_samples: int = 5, n_features: int = 2) -> np.ndarray:
    """Reproducible feature matrix already inside the [0, 1) range."""
    return _rng().random((n_samples, n_features))


class TestConstruction:
    """Constructor wiring and default attribute state."""

    def test_default_construction(self) -> None:
        det = QuantumAnomalyDetector()
        assert isinstance(det._executor, QuantumExecutor)
        assert det._shots == 1024
        assert det._error_mitigation == "zne"
        assert isinstance(det._error_mitigator, ErrorMitigationCircuit)
        assert det._trained_model is None
        assert det._method is None
        assert det._num_qubits == 4
        assert det._threshold == 0.5

    def test_error_mitigation_none_leaves_mitigator_unset(self) -> None:
        det = QuantumAnomalyDetector(error_mitigation=None)
        assert det._error_mitigator is None
        assert det._error_mitigation is None

    def test_error_mitigation_pec_builds_mitigator(self) -> None:
        det = QuantumAnomalyDetector(error_mitigation="pec")
        assert isinstance(det._error_mitigator, ErrorMitigationCircuit)

    def test_custom_shots_propagate(self) -> None:
        det = QuantumAnomalyDetector(shots=256)
        assert det._shots == 256

    def test_supported_methods_constant(self) -> None:
        assert QuantumAnomalyDetector.SUPPORTED_METHODS == [
            "quantum_kernel",
            "vqe_anomaly",
            "qaoa_anomaly",
            "amplitude_estimation",
        ]


class TestQuantumDetectionResult:
    """The detection result dataclass and its defaults."""

    def test_defaults(self) -> None:
        scores = np.array([0.1, 0.9])
        preds = np.array([0, 1])
        result = QuantumDetectionResult(
            anomaly_scores=scores,
            predictions=preds,
            threshold=0.5,
            method="vqe_anomaly",
        )
        assert result.quantum_metrics == {}
        assert result.classical_fallback_used is False
        assert result.details == {}
        np.testing.assert_array_equal(result.anomaly_scores, scores)
        np.testing.assert_array_equal(result.predictions, preds)

    def test_explicit_fields(self) -> None:
        result = QuantumDetectionResult(
            anomaly_scores=np.array([0.5]),
            predictions=np.array([1]),
            threshold=0.3,
            method="quantum_kernel",
            quantum_metrics={"num_qubits": 4, "shots": 1024},
            classical_fallback_used=True,
            details={"reason": "test"},
        )
        assert result.method == "quantum_kernel"
        assert result.threshold == 0.3
        assert result.quantum_metrics == {"num_qubits": 4, "shots": 1024}
        assert result.classical_fallback_used is True
        assert result.details == {"reason": "test"}


class TestQuantumResourceEstimate:
    """The resource-estimate dataclass and its defaults."""

    def test_defaults(self) -> None:
        est = QuantumResourceEstimate(
            num_qubits=4,
            circuit_depth=12,
            num_gates=18,
            estimated_runtime_ms=1.0,
            estimated_shots=100,
            hardware_compatible=True,
        )
        assert est.recommendations == []

    def test_explicit_fields(self) -> None:
        est = QuantumResourceEstimate(
            num_qubits=8,
            circuit_depth=72,
            num_gates=96,
            estimated_runtime_ms=42.0,
            estimated_shots=1760,
            hardware_compatible=False,
            recommendations=["a", "b"],
        )
        assert est.recommendations == ["a", "b"]
        assert est.hardware_compatible is False


class TestEstimateResources:
    """Closed-form resource accounting is fully deterministic."""

    def test_quantum_kernel_exact(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((5, 2), method="quantum_kernel")
        assert est.num_qubits == 2
        assert est.circuit_depth == 8
        assert est.num_gates == 24
        # shots * n * (n + 1) // 2 == 32 * 5 * 6 // 2
        assert est.estimated_shots == 480
        assert est.estimated_runtime_ms == pytest.approx(48.0048)
        assert est.hardware_compatible is True
        assert est.recommendations == []

    def test_vqe_exact(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((5, 2), method="vqe_anomaly")
        assert est.circuit_depth == 12
        assert est.num_gates == 18
        assert est.estimated_shots == 160
        assert est.estimated_runtime_ms == pytest.approx(16.0036)

    def test_qaoa_exact(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((5, 2), method="qaoa_anomaly")
        assert est.circuit_depth == 18
        assert est.num_gates == 18
        assert est.estimated_shots == 160

    def test_amplitude_estimation_uses_else_branch(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((5, 2), method="amplitude_estimation")
        assert est.circuit_depth == 8  # num_qubits * 4
        assert est.num_gates == 12  # num_qubits * 6
        assert est.estimated_shots == 160

    def test_unknown_method_uses_else_branch(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((5, 2), method="not_a_real_method")
        assert est.circuit_depth == 8
        assert est.num_gates == 12

    def test_default_method_is_vqe(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((5, 2))
        # vqe depth differs from the else-branch, confirming the default.
        assert est.circuit_depth == 12
        assert est.num_gates == 18

    def test_one_dimensional_shape(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((7,), method="vqe_anomaly")
        assert est.num_qubits == 1  # min(8, 1)
        assert est.circuit_depth == 6
        assert est.num_gates == 8
        assert est.estimated_shots == 224

    def test_qubit_count_recommendation(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((5, 10), method="quantum_kernel")
        assert est.num_qubits == 8  # min(8, 10)
        assert "Consider feature selection to reduce qubit count" in est.recommendations

    def test_high_depth_recommendation(self) -> None:
        det = QuantumAnomalyDetector(shots=32)
        est = det.estimate_resources((5, 8), method="qaoa_anomaly")
        assert est.circuit_depth == 72  # > 50
        assert "High circuit depth may benefit from error mitigation" in est.recommendations

    def test_long_runtime_recommendation(self) -> None:
        det = QuantumAnomalyDetector()  # default shots=1024
        est = det.estimate_resources((1000, 2), method="vqe_anomaly")
        assert est.estimated_runtime_ms > 60000
        assert "Long runtime - consider batching or simulation" in est.recommendations

    def test_all_recommendations_combined(self) -> None:
        det = QuantumAnomalyDetector()
        est = det.estimate_resources((1000, 8), method="qaoa_anomaly")
        assert est.recommendations == [
            "Consider feature selection to reduce qubit count",
            "High circuit depth may benefit from error mitigation",
            "Long runtime - consider batching or simulation",
        ]

    def test_hardware_compatible_always_true_within_caps(self) -> None:
        # num_qubits is capped at 8 and the deepest ansatz (qaoa @ 8q) is 72,
        # so hardware_compatible can never fall to False here.
        det = QuantumAnomalyDetector()
        for method in QuantumAnomalyDetector.SUPPORTED_METHODS:
            est = det.estimate_resources((50, 8), method=method)
            assert est.hardware_compatible is True


class TestNormalizeData:
    """Min-max normalisation with the shared 1-D reshape path."""

    def test_one_dimensional_reshaped_to_column(self) -> None:
        det = QuantumAnomalyDetector()
        out = det._normalize_data(np.array([0.0, 5.0, 10.0]))
        assert out.shape == (3, 1)
        np.testing.assert_allclose(out.ravel(), [0.0, 0.5, 1.0])

    def test_two_dimensional_columnwise(self) -> None:
        det = QuantumAnomalyDetector()
        out = det._normalize_data(np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]]))
        np.testing.assert_allclose(out, [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    def test_constant_column_maps_to_zero(self) -> None:
        det = QuantumAnomalyDetector()
        out = det._normalize_data(np.array([[3.0, 3.0], [3.0, 3.0]]))
        np.testing.assert_allclose(out, np.zeros((2, 2)))

    def test_output_stays_in_unit_range(self) -> None:
        det = QuantumAnomalyDetector()
        out = det._normalize_data(_rng().random((6, 3)) * 100 - 50)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestNotFittedClassicalFallback:
    """Un-fitted quantum methods route to the classical z-score detector."""

    def test_exact_scores_and_predictions(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        data = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        result = det.detect(data, method="vqe_anomaly")
        assert result.method == "classical_zscore"
        assert result.classical_fallback_used is True
        assert result.details == {"reason": "Classical fallback used"}
        np.testing.assert_allclose(result.anomaly_scores, [0.5, 0.0, 1.0])
        np.testing.assert_array_equal(result.predictions, [0, 0, 1])

    def test_classical_fallback_for_quantum_kernel_and_qaoa(self) -> None:
        data = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        for method in ("quantum_kernel", "qaoa_anomaly"):
            det = QuantumAnomalyDetector(shots=FAST_SHOTS)
            result = det.detect(data, method=method)
            assert result.method == "classical_zscore"
            assert result.classical_fallback_used is True

    def test_one_dimensional_input_reshaped(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        result = det.detect(np.array([0.0, 1.0, 2.0, 10.0]), method="quantum_kernel")
        assert result.anomaly_scores.shape == (4,)
        assert result.predictions.shape == (4,)
        assert result.method == "classical_zscore"

    def test_predictions_are_binary_int(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        result = det.detect(_sample_matrix(), method="vqe_anomaly")
        assert set(np.unique(result.predictions)).issubset({0, 1})
        assert np.issubdtype(result.predictions.dtype, np.integer)


class TestFit:
    """Training dispatch across every supported method."""

    def test_returns_self_for_chaining(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        returned = det.fit(_sample_matrix(), method="vqe_anomaly", num_qubits=FAST_QUBITS)
        assert returned is det
        assert det._method == "vqe_anomaly"

    def test_quantum_kernel_builds_model_dict(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="quantum_kernel", num_qubits=FAST_QUBITS)
        assert isinstance(det._trained_model, dict)
        assert set(det._trained_model) == {"kernel", "predict", "X_train"}
        assert callable(det._trained_model["predict"])

    def test_quantum_kernel_accepts_labels(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        X = _sample_matrix(n_samples=4, n_features=2)
        y = np.array([1.0, 1.0, -1.0, -1.0])
        det.fit(X, y_train=y, method="quantum_kernel", num_qubits=FAST_QUBITS)
        assert isinstance(det._trained_model, dict)

    def test_vqe_builds_detector(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="vqe_anomaly", num_qubits=FAST_QUBITS)
        assert isinstance(det._trained_model, VQEAnomalyDetector)

    def test_qaoa_builds_detector(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="qaoa_anomaly", num_qubits=FAST_QUBITS, maxiter=3)
        assert isinstance(det._trained_model, QAOAAnomalyDetector)

    def test_unknown_method_falls_back_to_vqe(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="mystery", num_qubits=FAST_QUBITS)
        assert isinstance(det._trained_model, VQEAnomalyDetector)
        assert det._method == "mystery"

    def test_num_qubits_defaults_to_min_four_features(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(n_samples=4, n_features=7), method="vqe_anomaly", maxiter=2)
        assert det._num_qubits == 4  # min(4, 7)

    def test_num_qubits_matches_features_when_below_four(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(n_samples=4, n_features=3), method="vqe_anomaly", maxiter=2)
        assert det._num_qubits == 3

    def test_num_qubits_explicit_override(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(n_samples=4, n_features=5), method="vqe_anomaly", num_qubits=2)
        assert det._num_qubits == 2

    def test_one_dimensional_training_data(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(
            _rng().random(6),
            method="vqe_anomaly",
            num_qubits=FAST_QUBITS,
            maxiter=2,
        )
        assert isinstance(det._trained_model, VQEAnomalyDetector)


def _assert_quantum_result(result: QuantumDetectionResult, method: str, n: int) -> None:
    """Sampling-invariant contract shared by every quantum pipeline."""
    assert isinstance(result, QuantumDetectionResult)
    assert result.method == method
    assert result.classical_fallback_used is False
    assert result.anomaly_scores.shape == (n,)
    assert result.predictions.shape == (n,)
    assert np.issubdtype(result.predictions.dtype, np.integer)
    assert set(np.unique(result.predictions)).issubset({0, 1})
    assert np.all(np.isfinite(result.anomaly_scores))
    assert np.all(result.anomaly_scores >= 0.0)
    assert set(result.quantum_metrics) == {"num_qubits", "shots"}
    assert result.quantum_metrics["shots"] == FAST_SHOTS


class TestDetectQuantumPipelines:
    """End-to-end fit + detect for each real quantum method.

    The NumPy sampler seeds from OS entropy, so exact scores are not
    reproducible; the assertions target the sampling-invariant contract
    (shape / dtype / sign / finiteness / metadata) which holds every run.
    """

    def test_vqe_pipeline(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="vqe_anomaly", num_qubits=FAST_QUBITS)
        test = _sample_matrix(n_samples=3, n_features=2)
        result = det.detect(test)
        _assert_quantum_result(result, "vqe_anomaly", 3)
        assert result.quantum_metrics["num_qubits"] == FAST_QUBITS

    def test_qaoa_pipeline(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="qaoa_anomaly", num_qubits=FAST_QUBITS, maxiter=3)
        result = det.detect(_sample_matrix(n_samples=3, n_features=2))
        _assert_quantum_result(result, "qaoa_anomaly", 3)

    def test_quantum_kernel_pipeline(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="quantum_kernel", num_qubits=FAST_QUBITS)
        result = det.detect(_sample_matrix(n_samples=3, n_features=2))
        _assert_quantum_result(result, "quantum_kernel", 3)
        # kernel-derived anomaly scores live in [0, 1].
        assert np.all(result.anomaly_scores <= 1.0)

    def test_amplitude_estimation_without_fit(self) -> None:
        # amplitude_estimation is not in the "needs a fitted model" set, so it
        # runs the quantum path directly even on a fresh detector.
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        result = det.detect(
            _sample_matrix(n_samples=3, n_features=2), method="amplitude_estimation"
        )
        _assert_quantum_result(result, "amplitude_estimation", 3)
        assert result.quantum_metrics["num_qubits"] == 4  # detector default

    def test_detect_reuses_fitted_method_when_unspecified(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="qaoa_anomaly", num_qubits=FAST_QUBITS, maxiter=3)
        result = det.detect(_sample_matrix(n_samples=2, n_features=2))
        assert result.method == "qaoa_anomaly"

    def test_unknown_detect_method_runs_vqe_path(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det.fit(_sample_matrix(), method="vqe_anomaly", num_qubits=FAST_QUBITS)
        result = det.detect(_sample_matrix(n_samples=2, n_features=2), method="mystery")
        # method label echoes the request; scoring used the VQE fallback branch.
        assert result.method == "mystery"
        assert result.anomaly_scores.shape == (2,)


class TestDetectErrorHandling:
    """Failure of the quantum path and its fallback / re-raise contract."""

    def _broken_detector(self) -> QuantumAnomalyDetector:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        # A non-None model that lacks ``.score`` makes the VQE scoring path
        # raise AttributeError once detection is attempted.
        det._trained_model = object()
        det._method = "vqe_anomaly"
        return det

    def test_failure_triggers_classical_fallback(self) -> None:
        det = self._broken_detector()
        data = np.array([[0.1, 0.2], [0.9, 0.8]])
        result = det.detect(data, classical_fallback=True)
        assert result.classical_fallback_used is True
        assert result.method == "classical_zscore"

    def test_failure_reraises_without_fallback(self) -> None:
        det = self._broken_detector()
        data = np.array([[0.1, 0.2], [0.9, 0.8]])
        with pytest.raises(AttributeError):
            det.detect(data, classical_fallback=False)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_empty_input_raises_value_error(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        with pytest.raises(ValueError):
            det.detect(np.empty((0, 2)), method="vqe_anomaly")


class TestAmplitudeEstimationInternals:
    """Exercise the list-batch unwrap and zero-probability skip in the
    amplitude-estimation scorer with a deterministic stubbed executor."""

    def test_list_result_unwrapped_and_zero_prob_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        det._num_qubits = 2  # uniform_prob == 1 / 2**2 == 0.25

        # A batch-style (list) result forces the ``result = result[0]``
        # unwrap; the 0.0 outcome forces the ``prob > 0`` false branch.
        stub_result = SimpleNamespace(probabilities={"00": 1.0, "01": 0.0})

        class _StubExecutor:
            def run(self, circuit: object) -> list[object]:
                return [stub_result]

        monkeypatch.setattr(det, "_executor", _StubExecutor())
        result = det.detect(np.array([[0.2, 0.8]]), method="amplitude_estimation")

        assert result.method == "amplitude_estimation"
        assert result.anomaly_scores.shape == (1,)
        # KL vs uniform: 1.0 * log(1.0 / 0.25) == log(4); the 0.0 term is skipped.
        np.testing.assert_allclose(result.anomaly_scores, [np.log(4.0)], rtol=1e-6)


class TestThresholdAndMethodResolution:
    """Threshold defaulting quirks and method resolution order."""

    def test_explicit_zero_threshold_is_respected(self) -> None:
        # Regression: ``threshold or self._threshold`` treated an explicit 0.0
        # as falsy and silently replaced it with the 0.5 default. The None-check
        # now preserves 0.0 -- a legitimate "flag anything scoring above zero"
        # threshold.
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        result = det.detect(_sample_matrix(), method="vqe_anomaly", threshold=0.0)
        assert result.threshold == 0.0

    def test_custom_threshold_is_applied(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        data = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        result = det.detect(data, method="vqe_anomaly", threshold=0.3)
        assert result.threshold == 0.3
        # classical scores are [0.5, 0.0, 1.0]; only the first & last exceed 0.3.
        np.testing.assert_array_equal(result.predictions, [1, 0, 1])

    def test_method_defaults_to_vqe_when_unset(self) -> None:
        det = QuantumAnomalyDetector(shots=FAST_SHOTS)
        # No fitted method and no explicit method -> "vqe_anomaly" is chosen,
        # which (un-fitted) routes to the classical fallback.
        result = det.detect(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert result.method == "classical_zscore"
        assert result.classical_fallback_used is True
