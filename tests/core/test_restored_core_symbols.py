# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restored ``core`` symbols removed in 7ab59916 and f498866a as uncalled.

Each is pinned to its released contract; the defect tests fail on the
released code, where noted: a detector overrunning the executor's timeout
discarded every finished result, ``compute_benchmark_metrics`` ignored
``domain``, ``_safe_cholesky`` raised on any matrix with an eigenvalue below
``-tikhonov_lambda``, the legacy OAE optimizer returned its initial weights,
and unknown ``mode`` / ``method`` strings were silently replaced.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core import three_r_mechanism as trm
from omni_mercury_engine.core.base_domains import (
    BaseDomainDetector,
    ParallelDetectorExecutor,
    create_enhanced_detector,
)
from omni_mercury_engine.core.domain_metrics import compute_benchmark_metrics
from omni_mercury_engine.core.governed_fusion import precision_sqrt
from omni_mercury_engine.core.robust_pooling import reliability_pool
from omni_mercury_engine.detectors.info_geometry import _safe_cholesky


class _Detector:
    parallel_safe = True

    def __init__(self, delay: float, fail: bool = False) -> None:
        self.delay, self.fail = delay, fail

    def detect(self, x: np.ndarray) -> float:
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("bad input")
        return float(np.sum(x))


class TestParallelDetectorExecutor:
    def test_overrun_keeps_finished_results_and_returns_at_the_deadline(self) -> None:
        """Fails on the released executor, which raised TimeoutError after 2 s."""
        executor = ParallelDetectorExecutor(max_workers=4, timeout=0.5)
        detectors = {"slow": _Detector(2.0), "fast": _Detector(0.0), "bad": _Detector(0.0, True)}
        start = time.perf_counter()
        results = executor.execute_detectors(detectors, np.ones(3))
        assert time.perf_counter() - start < 1.5
        assert list(results) == ["slow", "fast", "bad"]
        assert results["fast"] == 3.0
        assert results["slow"] == {"error": "timeout after 0.5 s"}
        assert results["bad"] == {"error": "bad input"}

    def test_detectors_without_the_method_are_left_out(self) -> None:
        executor = ParallelDetectorExecutor()
        results = executor.execute_detectors({"a": _Detector(0.0), "b": object()}, np.ones(2))
        assert results == {"a": 2.0}

    def test_undeclared_detectors_run_in_the_calling_thread(self) -> None:
        import threading

        class Unsafe:
            def detect(self, _x: np.ndarray) -> str:
                return threading.current_thread().name

        results = ParallelDetectorExecutor().execute_detectors(
            {"u": Unsafe(), "v": Unsafe()}, np.ones(1)
        )
        assert set(results.values()) == {threading.current_thread().name}


def test_create_enhanced_detector_wraps_the_base_detector() -> None:
    class Base:
        def __init__(self, config: dict[str, Any] | None) -> None:
            self.config = config

    wrapped = create_enhanced_detector(Base, "temporal", {"k": 1})
    assert isinstance(wrapped, BaseDomainDetector)
    assert wrapped.domain == "temporal"
    assert wrapped.base_detector.config == {"k": 1}


class TestComputeBenchmarkMetrics:
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])

    def test_temporal_computes_event_metrics(self) -> None:
        """Fails on the released function, which ignored ``domain``."""
        general = compute_benchmark_metrics(self.y_true, self.y_pred)
        temporal = compute_benchmark_metrics(self.y_true, self.y_pred, domain="temporal")
        assert general.event_f1 == 0.0  # not computed for 8 samples without timestamps
        assert temporal.event_f1 == 1.0

    def test_graph_uses_the_adjacency_matrix(self) -> None:
        n = self.y_true.size
        adjacency = np.ones((n, n)) - np.eye(n)
        metrics = compute_benchmark_metrics(
            self.y_true,
            self.y_pred,
            np.linspace(0.0, 1.0, n),
            domain="graph",
            adjacency_matrix=adjacency,
        )
        assert metrics.morans_i == pytest.approx(-1.0 / (n - 1))  # complete-graph value

    @pytest.mark.parametrize("domain", ["dimensional", "statistical", "medical"])
    def test_other_domains_get_the_standard_suite(self, domain: str) -> None:
        general = compute_benchmark_metrics(self.y_true, self.y_pred)
        other = compute_benchmark_metrics(self.y_true, self.y_pred, domain=domain)
        assert other.to_dict() == general.to_dict()

    def test_spatial_without_weights_says_what_is_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            metrics = compute_benchmark_metrics(self.y_true, self.y_pred, domain="spatial")
        assert metrics.morans_i is None
        assert "spatial_weights" in caplog.text


def test_reliability_pool_dispatches_and_rejects_unknown_modes() -> None:
    probs = np.array([[0.9, 0.2, 0.99]])
    weights = np.ones(3)
    assert reliability_pool(probs, weights)[0] == pytest.approx(0.9)  # outliers clipped
    assert reliability_pool(probs, weights, mode="trimmed")[0] == pytest.approx(
        1.0 / (1.0 + np.exp(-(np.log(9.0) + np.log(99.0)) / 2.0))
    )
    with pytest.raises(ValueError, match="mode"):  # released: silently clipped
        reliability_pool(probs, weights, mode="median")


def test_precision_sqrt_is_a_symmetric_root() -> None:
    precision = np.array([[2.0, 0.5], [0.5, 1.0]])
    root = precision_sqrt(precision)
    np.testing.assert_allclose(root, root.T)
    np.testing.assert_allclose(root.T @ root, precision, atol=1e-12)
    indefinite = precision_sqrt(np.array([[1.0, 0.0], [0.0, -1.0]]))
    np.testing.assert_allclose(indefinite, np.diag([1.0, 0.0]), atol=1e-12)


class TestSafeCholesky:
    def test_positive_definite_is_unshifted(self) -> None:
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        lower = _safe_cholesky(matrix)
        np.testing.assert_allclose(lower @ lower.T, matrix)

    def test_indefinite_matrix_gets_a_near_minimal_shift(self) -> None:
        """Fails on the released single 1e-6 shift (LinAlgError)."""
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
        lower = _safe_cholesky(matrix)
        shift = np.diag(lower @ lower.T - matrix)
        assert np.allclose(shift, shift[0])
        assert 1.0 < shift[0] <= 2.0 + 1e-6  # within a factor of two of the minimum

    def test_non_finite_matrix_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _safe_cholesky(np.array([[np.nan, 0.0], [0.0, 1.0]]))


class TestLegacyThreeR:
    def test_identical_names_resolve_to_the_live_classes(self) -> None:
        for name in (
            "RecursionEngine",
            "ResonanceEngine",
            "AnomalyDetectionMethod",
            "IssueType",
            "IssueSeverity",
            "EvolutionStrategy",
        ):
            assert getattr(trm, f"_Legacy{name}") is getattr(trm, name)

    def test_legacy_equation_keeps_its_released_behaviour(self) -> None:
        """The live equation drops eta**Phi and uses other default weights."""
        equation = trm._LegacyOmniAvaEquation()
        assert equation.weights == pytest.approx(
            {"w_R": 0.5, "w_H": 0.3090169943749474, "w_O": 0.19098300562505255}
        )
        result = equation.compute(0.8, 0.6, 0.4)
        assert result.fusion_score == pytest.approx(0.6195027602420753)  # released value
        assert "baseline_comparison" in equation.get_dominance_proof()

    def test_legacy_refactoring_config_keeps_its_field_order(self) -> None:
        config = trm._LegacyRefactoringConfig(True, False)
        assert (config.apply_refactorings, config.create_backup) == (True, False)

    @staticmethod
    def _data(seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y = (rng.random(400) < 0.2).astype(int)
        signal = np.clip(0.3 + 0.5 * y + rng.normal(0.0, 0.1, 400), 0.0, 1.0)
        return np.c_[rng.random(400), rng.random(400), signal], y

    def test_f1_fit_moves_off_the_initial_weights(self) -> None:
        """Fails on the released SLSQP fit, which stopped after one iteration."""
        X, y = self._data(0)
        optimizer = trm._LegacyOAEWeightOptimizer(domain="medical")
        result = optimizer.optimize(X, y)
        assert result["baseline_f1"] < 0.4
        assert result["optimized_f1"] > 0.95
        assert sum(result["weights"].values()) == pytest.approx(1.0)
        assert result["weights"]["w_O"] > 0.9
        X_test, y_test = self._data(1)
        assert optimizer.optimized_weights is not None
        held_out = -optimizer._objective_function(optimizer.optimized_weights, X_test, y_test)
        assert held_out > 0.95

    def test_get_optimized_equation(self) -> None:
        optimizer = trm._LegacyOAEWeightOptimizer()
        assert optimizer.get_optimized_equation() is None
        X, y = self._data(0)
        optimizer.optimize(X, y)
        equation = optimizer.get_optimized_equation()
        assert isinstance(equation, trm.OmniAvaEquation)
        assert optimizer.optimized_weights is not None
        assert equation.weights["w_O"] == pytest.approx(float(optimizer.optimized_weights[2]))

    def test_input_shape_is_validated(self) -> None:
        optimizer = trm._LegacyOAEWeightOptimizer()
        with pytest.raises(ValueError, match="3 columns"):
            optimizer.optimize(np.zeros((4, 2)), np.zeros(4))
        with pytest.raises(ValueError, match="y must have shape"):
            optimizer.optimize(np.zeros((4, 3)), np.zeros(3))


def test_threshold_defaults_and_load_configuration(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omni_mercury_engine.core import config
    from omni_mercury_engine.core.config import ThresholdDefaults, load_configuration

    monkeypatch.setattr(config, "_config_manager", None)  # a fresh, discarded singleton
    assert ThresholdDefaults.ANOMALY_CLASSIFICATION_DEFAULT == 0.5
    assert ThresholdDefaults.SIGMA_DIRECTIVE_WEIGHTED_SCORE == 0.8
    path = tmp_path / "mercury.json"
    path.write_text('{"restored_probe": {"value": 7}}')
    manager = load_configuration([path, tmp_path / "absent.yaml"], load_env=False)
    assert manager.get("restored_probe.value") == 7


def test_add_conformal_to_detector_holds_its_coverage() -> None:
    from omni_mercury_engine.core.conformal_prediction import add_conformal_to_detector

    class Distance:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.asarray(1.0 - np.exp(-np.linalg.norm(X, axis=1)))

    rng = np.random.default_rng(0)
    detector = Distance()
    _, threshold = add_conformal_to_detector(detector, rng.normal(size=(2000, 3)), 0.95)
    exceed = float(np.mean(detector.predict_proba(rng.normal(size=(4000, 3))) > threshold))
    assert 0.03 < exceed < 0.07
    with pytest.raises(ValueError, match="method"):  # released: silently cross-conformal
        add_conformal_to_detector(detector, rng.normal(size=(50, 3)), method="jackknife")


def test_optimize_benevolent_detector_returns_bounded_parameters() -> None:
    from omni_mercury_engine.core.benevolence_optimization import optimize_benevolent_detector

    class Threshold:
        def __init__(self, params: np.ndarray) -> None:
            self.scale = float(params[0])

        def fit(self, _X: np.ndarray, _y: np.ndarray) -> None:
            return None

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-self.scale * (X[:, 0] - 1.0)))

    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3).astype(int)
    X = np.c_[rng.normal(2.0 * y, 0.5), rng.normal(size=200)]
    params, front = optimize_benevolent_detector(
        Threshold, X, y, [(0.1, 10.0)], n_generations=3, seed=0
    )
    assert front.solutions
    assert 0.1 <= float(params[0]) <= 10.0


def test_optimize_gosnn_runs_the_optimizer() -> None:
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork
    from omni_mercury_engine.core.gosnn_optimizer import OptimizationResult, optimize_gosnn

    result = optimize_gosnn(GlobalOmniScalarNetwork(), sigma_immutable=0.96)
    assert isinstance(result, OptimizationResult)
    assert result.total_scalars > 0


def test_create_enhanced_model() -> None:
    from omni_mercury_engine.core.model_domains import (
        AffectiveStateModel,
        MultimodalBiometricModel,
        QuantumFeatureModel,
        create_enhanced_model,
    )

    assert isinstance(create_enhanced_model("quantum"), QuantumFeatureModel)
    assert isinstance(create_enhanced_model("biometric"), MultimodalBiometricModel)
    assert isinstance(create_enhanced_model("affective"), AffectiveStateModel)
    with pytest.raises(ValueError, match="Unknown model type"):
        create_enhanced_model("tabular")


def test_run_baseline_benchmarks_scores_separable_data() -> None:
    from omni_mercury_engine.core.rigorous_benchmark import run_baseline_benchmarks

    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0.0, 1.0, (180, 4)), rng.normal(4.0, 1.0, (20, 4))]
    y = np.r_[np.zeros(180), np.ones(20)].astype(int)
    results = run_baseline_benchmarks(X, y, n_folds=3)
    assert list(results) == ["Mercury-Agent"]
    assert results["Mercury-Agent"].roc_auc.mean > 0.9  # 0.98 measured


def test_engine_restored_attributes() -> None:
    from omni_mercury_engine.core.ethical_config import EngineConfig
    from omni_mercury_engine.engine import OmniMercuryEngine

    assert EngineConfig().enable_ci is False
    assert OmniMercuryEngine.MAX_WORKERS == 4
    engine = OmniMercuryEngine()
    workers = max(1, int(engine.config.num_workers))
    assert engine.optimization_config.n_jobs == workers
    assert engine.parallel_executor.n_jobs == workers
    assert engine.parallel_executor.map(abs, [-1, 2, -3]) == [1, 2, 3]
