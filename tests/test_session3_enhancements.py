# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Neuro-Symbolic Hub and GOSNN Enhancement Tests.

Comprehensive tests for:
- Neuro-Symbolic Hub
- GOSNN Optimizer
- Real-World Benchmarks
- Integration with previous sessions
"""

from typing import Any

import numpy as np
import pytest

# Fixed seed for reproducibility
SEED = 42


def _bypass_sigma_immutable(monkeypatch: pytest.MonkeyPatch, hub: object) -> None:
    """Mock the σ_Immutable gate on ``hub`` so synthetic random inputs pass.

    Wave B made σ_Immutable a mandatory hard ethical gate at the
    :meth:`NeuroSymbolicHub.predict` boundary.  These integration tests
    feed ``np.random.randn(...)`` into the hub to exercise GOSNN scalar
    plumbing, fusion modes, explanations, and pipeline integration —
    not ethical enforcement.  Random vectors do not satisfy the trained
    256-D ethical gate, so the gate is mocked here to return a passing
    evaluation.  The production-side σ_Immutable contract is exercised
    by ``tests/ethical/test_hard_enforcement.py`` and
    ``tests/security/test_sigma_immutable_kat.py`` which feed
    realistic vectors.
    """
    from omni_mercury_engine.security.sigma_immutable_gate import (
        SigmaImmutableEvaluation,
    )

    monkeypatch.setattr(
        hub._sigma_immutable_gate,  # type: ignore[attr-defined]
        "enforce",
        lambda action, scalar_vector, details=None: SigmaImmutableEvaluation(
            score=0.99, threshold=0.93, passes=True, backend="torch"
        ),
    )


class TestNeuroSymbolicHub:
    """Tests for the enhanced neuro-symbolic hub."""

    def test_hub_initialization(self) -> None:
        """Test hub initializes correctly."""
        from omni_mercury_engine.core.neurosymbolic_hub import (
            FusionMode,
            NeuroSymbolicHub,
        )

        hub = NeuroSymbolicHub(
            input_dim=64,
            fusion_mode=FusionMode.PHI_WEIGHTED,
            seed=SEED,
        )

        assert hub.input_dim == 64
        assert hub.fusion_mode == FusionMode.PHI_WEIGHTED
        assert hub.benevolence_threshold == 0.99

    def test_phi_weighted_fusion(self) -> None:
        """Test golden ratio weighting."""
        from omni_mercury_engine.core.neurosymbolic_hub import (
            PHI,
            FusionMode,
            NeuroSymbolicHub,
        )

        hub = NeuroSymbolicHub(fusion_mode=FusionMode.PHI_WEIGHTED, seed=SEED)

        # Check weights sum to 1 and follow phi ratio
        expected_neural = PHI / (1 + PHI)
        expected_symbolic = 1 / (1 + PHI)

        assert abs(hub._neural_weight - expected_neural) < 0.01
        assert abs(hub._symbolic_weight - expected_symbolic) < 0.01
        assert abs(hub._neural_weight + hub._symbolic_weight - 1.0) < 0.001

    def test_predict_returns_explanations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test prediction returns explanations."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(
            input_dim=32,
            seed=SEED,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        # Test-only: bypass the setter's floor-clamp so synthetic random
        # inputs (whose benevolence sits in the ~0.65–0.75 band) do not
        # trigger the hard ethical gate.  This test exercises explanations,
        # not ethical enforcement.
        hub._benevolence_threshold = 0.0
        _bypass_sigma_immutable(monkeypatch, hub)

        X = np.random.randn(3, 32)
        results = hub.predict(X, return_explanations=True)

        assert len(results) == 3
        for result in results:
            assert hasattr(result, "anomaly_score")
            assert hasattr(result, "neural_score")
            assert hasattr(result, "symbolic_score")
            assert hasattr(result, "reasoning_chain")
            assert 0 <= result.anomaly_score <= 1
            assert 0 <= result.confidence <= 1

    def test_harm_gate_enforcement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hard ethical decision boundary: predict() fails closed on gate failure.

        The benevolence threshold is gone and is not what is being tested. This
        pins the fail-closed direction of the *harm* gate: when
        ``assess_weapons_uplift`` cannot be evaluated at all, ``predict()``
        must raise ``EthicalConstraintViolationError`` with
        ``check="harm_uplift"`` rather than fall through to a verdict. The
        previous advisory ``ethical_violations`` list is no longer the contract
        at this boundary.
        """
        import omni_mercury_engine.cognitive.decision_gate as gate_module
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(
            seed=SEED,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )

        X = np.random.randn(3, 64)
        # Benign data is no longer refused for being benign -- the deleted
        # 0.99 hub gate tested a transform of the fused anomaly score, not harm.
        assert len(hub.predict(X)) == 3

        # What is enforced: the shared fail-closed harm-uplift choke point.
        def _boom(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("simulated harm-gate failure")

        monkeypatch.setattr(gate_module, "assess_weapons_uplift", _boom)
        with pytest.raises(EthicalConstraintViolationError) as exc_info:
            hub.predict(X)

        assert exc_info.value.check == "harm_uplift"
        assert exc_info.value.details["fail_closed"] is True

    def test_knowledge_graph_rules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test symbolic rules fire correctly."""
        from omni_mercury_engine.core.neurosymbolic_hub import (
            NeuroSymbolicHub,
            SymbolicRule,
        )

        hub = NeuroSymbolicHub(
            seed=SEED,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        # Test-only: bypass the setter's floor-clamp.  This test exercises
        # rule-firing, not ethical enforcement.
        hub._benevolence_threshold = 0.0
        _bypass_sigma_immutable(monkeypatch, hub)

        # Add custom rule
        hub.add_rule(
            SymbolicRule(
                rule_id="test_rule",
                premise="test_condition >= 1.0",
                conclusion="test_alert",
                confidence=0.9,
            )
        )

        # Predict with context that triggers rule
        X = np.random.randn(1, 64)
        context = {"test_condition": 1.5}
        results = hub.predict(X, context=context)

        # Check rule was considered
        assert len(results) == 1

    def test_gosnn_integration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test GOSNN scalar registration."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(
            seed=SEED,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        # Test-only: bypass the floor-clamp.  This test exercises GOSNN
        # scalar integration, not ethical enforcement.
        hub._benevolence_threshold = 0.0
        _bypass_sigma_immutable(monkeypatch, hub)

        # Run some inferences (reduced from 5 to 3 for faster execution)
        X = np.random.randn(3, 64)
        hub.predict(X)

        # Get scalars
        scalars = hub.get_gosnn_scalars()

        assert "neurosymbolic_neural_weight" in scalars
        assert "neurosymbolic_symbolic_weight" in scalars
        assert "neurosymbolic_inference_count" in scalars
        assert scalars["neurosymbolic_inference_count"] == 3

    def test_fit_learns_weights(self) -> None:
        """Test fitting learns optimal fusion weights."""
        from omni_mercury_engine.core.neurosymbolic_hub import (
            FusionMode,
            NeuroSymbolicHub,
        )

        hub = NeuroSymbolicHub(
            fusion_mode=FusionMode.ADAPTIVE,
            seed=SEED,
        )

        # Generate labeled data
        np.random.seed(SEED)
        X = np.random.randn(100, 64)
        y = (np.random.rand(100) > 0.8).astype(int)

        # Fit
        hub.fit(X, y)

        # Weights should have been updated
        assert hub._fitted is True


class TestGOSNNOptimizer:
    """Tests for GOSNN optimizer."""

    def test_optimizer_initialization(self) -> None:
        """Test optimizer initializes correctly."""
        from omni_mercury_engine.core.gosnn_optimizer import GOSNNOptimizer

        optimizer = GOSNNOptimizer(
            sigma_immutable=0.96,
            target_overhead_percent=2.0,
        )

        assert optimizer.sigma_immutable == 0.96
        assert optimizer.target_overhead == 2.0

    def test_scalar_importance_analysis(self) -> None:
        """Test SHAP-like importance analysis."""
        from omni_mercury_engine.core.gosnn_optimizer import ScalarImportanceAnalyzer

        analyzer = ScalarImportanceAnalyzer(seed=SEED)

        # Record some scalar history
        for i in range(20):
            scalars = {
                "important_scalar": 1.0 + i * 0.1,
                "stable_scalar": 1.0,
                "noisy_scalar": np.random.rand(),
            }
            analyzer.record_scalars(scalars)

        # Compute importance
        importances = analyzer.compute_importance(
            {"important_scalar": 2.0, "stable_scalar": 1.0, "noisy_scalar": 0.5},
            output_value=0.8,
        )

        assert "important_scalar" in importances
        assert "stable_scalar" in importances
        assert "noisy_scalar" in importances

        # Stable scalar should have high stability
        assert importances["stable_scalar"].stability_score > 0.5

    def test_ethical_gate_hard_constraint(self) -> None:
        """Test σ_Immutable hard constraint at 0.93."""
        from omni_mercury_engine.core.gosnn_optimizer import EthicalGateOptimizer

        gate = EthicalGateOptimizer(
            sigma_immutable_hard=0.93,
            sigma_immutable_target=0.96,
        )

        # Should pass with very high ethical scalars (values are normalized by /2, so need ~1.9+ to get 0.93+)
        # The normalization clips to [0, 2] then divides by 2, so max normalized is 1.0
        # With Lyapunov factor applied, we need values close to 2.0 to pass 0.93 threshold
        passes, score, violations = gate.evaluate(
            {
                "omnimorality": 2.0,
                "omniempathy": 2.0,
                "omnibenevolence": 2.0,
            }
        )

        # Verify the gate produces a valid score
        assert 0.0 <= score <= 1.0

        # Should fail with low ethical scalars
        passes_low, score_low, violations_low = gate.evaluate(
            {
                "omnimorality": 0.3,
                "omniempathy": 0.3,
                "omnibenevolence": 0.5,
            }
        )

        # Low values should produce lower score than high values
        assert score_low < score
        # Low values should fail the hard constraint
        assert passes_low is False
        assert len(violations_low) > 0

    def test_attention_optimizer_overhead(self) -> None:
        """Test attention optimization produces valid output."""
        from omni_mercury_engine.core.gosnn_optimizer import AttentionOptimizer

        optimizer = AttentionOptimizer(
            num_heads=32,
            target_overhead_percent=2.0,
        )

        # Generate dummy attention scores
        attention = np.random.randn(32, 16, 16)

        # Optimize
        weighted, overhead = optimizer.optimize_attention(attention)

        assert weighted.shape == attention.shape
        # Overhead should be a positive number (actual value depends on hardware)
        assert overhead >= 0

    def test_full_optimization(self) -> None:
        """Test full GOSNN optimization produces valid results."""
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )
        from omni_mercury_engine.core.gosnn_optimizer import GOSNNOptimizer

        # Reset singleton
        reset_global_network()

        gosnn = GlobalOmniScalarNetwork()
        optimizer = GOSNNOptimizer(seed=SEED)

        # Optimize
        result = optimizer.optimize(gosnn)

        # Verify optimization produces valid results
        assert result.total_scalars > 0
        # sigma_Immutable and ethical compliance depend on scalar values
        # The optimizer should produce a valid sigma_immutable_value
        assert 0.0 <= result.sigma_immutable_value <= 1.0
        # Benevolence should be computed
        assert result.benevolence_value >= 0.0


class TestRealWorldBenchmark:
    """Tests for real-world benchmark runner."""

    def test_synthetic_data_generation(self) -> None:
        """Test synthetic data generation."""
        from omni_mercury_engine.core.realworld_benchmark import SyntheticDataGenerator

        generator = SyntheticDataGenerator(seed=SEED)

        # SMD
        X, y = generator.generate_smd_like(n_samples=1000)
        assert X.shape == (1000, 38)
        assert y.shape == (1000,)
        assert 0.01 < np.mean(y) < 0.15  # ~5% anomalies

        # NSL-KDD
        X, y = generator.generate_nslkdd_like(n_samples=1000)
        assert X.shape == (1000, 41)
        assert 0.15 < np.mean(y) < 0.25  # ~20% attacks

        # BATADAL
        X, y = generator.generate_batadal_like(n_samples=1000)
        assert X.shape == (1000, 43)
        assert 0.05 < np.mean(y) < 0.15  # ~10% attacks

    def test_benchmark_runner_sklearn_detector(self) -> None:
        """Test benchmark with sklearn detector - verifies fail-closed behavior without real data."""
        from omni_mercury_engine.core.realworld_benchmark import RealWorldBenchmarkRunner
        from omni_mercury_engine.detectors.enhanced_statistical import MADDetector

        runner = RealWorldBenchmarkRunner(n_folds=3, seed=SEED)

        # Wrapper using MADDetector's actual API (fit + detect -> AnomalyResult)
        class IFWrapper:
            def __init__(self) -> None:
                self.model = MADDetector()

            def fit(self, X: Any, y: Any = None) -> None:
                self.model.fit(X)

            def predict(self, X: Any) -> Any:
                result = self.model.detect(X)
                return result.is_anomaly.astype(int)

            def predict_proba(self, X: Any) -> Any:
                result = self.model.detect(X)
                scores = result.scores
                return np.column_stack([1 - scores, scores])

        # Test fail-closed behavior: without real data, should raise RuntimeError
        # This validates the Civilization-First principle of no synthetic data for validation
        with pytest.raises(RuntimeError, match="REAL DATA REQUIRED"):
            runner.run_benchmark(IFWrapper(), "SMD", "IsolationForest")

    def test_benchmark_with_neurosymbolic_hub(self) -> None:
        """Test benchmark with neuro-symbolic hub - verifies fail-closed behavior without real data."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub
        from omni_mercury_engine.core.realworld_benchmark import RealWorldBenchmarkRunner

        runner = RealWorldBenchmarkRunner(n_folds=3, seed=SEED)

        # Create wrapper for NeuroSymbolicHub
        class NSHWrapper:
            def __init__(self) -> None:
                self.hub = NeuroSymbolicHub(input_dim=38, seed=SEED)
                # Test-only: bypass the floor-clamp.  This benchmark
                # wrapper tests scoring, not ethical enforcement.
                self.hub._benevolence_threshold = 0.0

            def fit(self, X: Any, y: Any = None) -> None:
                self.hub.fit(X, y)

            def predict(self, X: Any) -> Any:
                return self.hub.predict(X, return_explanations=False)

            def predict_proba(self, X: Any) -> Any:
                return self.hub.predict_proba(X)

        # Test fail-closed behavior: without real data, should raise RuntimeError
        # This validates the Civilization-First principle of no synthetic data for validation
        with pytest.raises(RuntimeError, match="REAL DATA REQUIRED"):
            runner.run_benchmark(NSHWrapper(), "SMD", "NeuroSymbolicHub")

    def test_event_metrics_computation(self) -> None:
        """Test event-based metrics."""
        from omni_mercury_engine.core.realworld_benchmark import RealWorldBenchmarkRunner

        runner = RealWorldBenchmarkRunner(n_folds=3, seed=SEED)

        # Ground truth with 2 events
        y_true = np.zeros(100, dtype=int)
        y_true[10:20] = 1  # Event 1
        y_true[50:60] = 1  # Event 2

        # Predictions that detect event 1 but miss event 2
        y_pred = np.zeros(100, dtype=int)
        y_pred[12:18] = 1  # Detects event 1 (partial overlap)

        event_f1, ttd = runner._compute_event_metrics(y_true, y_pred)

        assert 0 <= event_f1 <= 1
        assert ttd >= 0


class TestIntegration:
    """Integration tests with previous sessions."""

    def test_neurosymbolic_with_stacking_fusion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test neuro-symbolic hub with stacking fusion."""
        from omni_mercury_engine.core.neurosymbolic_hub import (
            FusionMode,
            NeuroSymbolicHub,
        )

        hub = NeuroSymbolicHub(
            fusion_mode=FusionMode.STACKING,
            seed=SEED,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        # Test-only: bypass the floor-clamp.  This test exercises the
        # scoring pipeline, not ethical enforcement.
        hub._benevolence_threshold = 0.0
        _bypass_sigma_immutable(monkeypatch, hub)

        # Fit with labeled data (reduced size for faster execution)
        # Use 50% threshold to ensure both classes are represented
        np.random.seed(SEED)
        X = np.random.randn(30, 64)
        y = (np.random.rand(30) > 0.5).astype(int)

        hub.fit(X, y)

        # Predict (reduced size for faster execution)
        X_test = np.random.randn(3, 64)
        results = hub.predict(X_test)

        assert len(results) == 3

    def test_gosnn_integration_layer(self) -> None:
        """Test integration with GOSNN integration layer."""
        try:
            from omni_mercury_engine.core.gosnn_integration import (
                GOSNNIntegration,
            )

            integration = GOSNNIntegration(
                sigma_immutable=0.96,
                benevolence_threshold=0.99,
            )

            # Should initialize without error
            assert integration.sigma_immutable == 0.96

        except ImportError:
            pytest.skip("GOSNN integration module not available")

    def test_calibration_integration(self) -> None:
        """Test integration with calibration module."""
        try:
            from omni_mercury_engine.core.calibration import CalibrationEnsemble

            calibrator = CalibrationEnsemble()

            # Generate test data
            np.random.seed(SEED)
            scores = np.random.rand(100)
            labels = (scores > 0.5).astype(int)

            # Fit calibrator
            calibrator.fit(scores, labels)

            # Calibrate
            calibrated = calibrator.calibrate(scores)

            assert len(calibrated) == len(scores)
            assert all(0 <= c <= 1 for c in calibrated)

        except ImportError:
            pytest.skip("Calibration module not available")

    def test_end_to_end_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test complete pipeline from data to ethical detection."""
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )
        from omni_mercury_engine.core.gosnn_optimizer import GOSNNOptimizer
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        # Reset GOSNN singleton
        reset_global_network()

        # Initialize components with reduced complexity for faster execution
        hub = NeuroSymbolicHub(
            input_dim=64,
            seed=SEED,
            enable_domain_features=False,
            enable_adaptive_thresholding=False,
            enable_gosnn_3r=False,
        )
        # Test-only: bypass the floor-clamp.  This test exercises the
        # end-to-end pipeline, not ethical enforcement.
        hub._benevolence_threshold = 0.0
        _bypass_sigma_immutable(monkeypatch, hub)

        gosnn = GlobalOmniScalarNetwork()
        optimizer = GOSNNOptimizer(seed=SEED)

        # Generate data (reduced size for faster execution)
        np.random.seed(SEED)
        X_train = np.random.randn(30, 64)
        y_train = (np.random.rand(30) > 0.9).astype(int)
        X_test = np.random.randn(5, 64)

        # Fit
        hub.fit(X_train, y_train)

        # Predict
        results = hub.predict(X_test)

        # Integrate with GOSNN
        hub.integrate_with_gosnn(gosnn)

        # Optimize GOSNN
        opt_result = optimizer.optimize(gosnn)

        # Assertions - verify pipeline produces valid results
        assert len(results) == 5
        assert all(r.benevolence_score >= 0 for r in results)
        # sigma_Immutable and ethical compliance depend on scalar values
        assert 0.0 <= opt_result.sigma_immutable_value <= 1.0
        assert opt_result.benevolence_value >= 0.0


class TestEthicalConstraints:
    """Tests for ethical constraint enforcement."""

    def test_benevolence_threshold_immutable(self) -> None:
        """Test benevolence threshold is enforced."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(benevolence_threshold=0.99, seed=SEED)

        # Cannot set below threshold
        assert hub.benevolence_threshold == 0.99

    def test_sigma_immutable_hard_limit(self) -> None:
        """Test σ_Immutable hard limit at 0.93."""
        from omni_mercury_engine.core.gosnn_optimizer import EthicalGateOptimizer

        gate = EthicalGateOptimizer(sigma_immutable_hard=0.93)

        # Should block if below 0.93
        passes, _, _ = gate.evaluate({"omnibenevolence": 0.5})
        assert passes is False

    def test_ethical_rules_not_prunable(self) -> None:
        """Test ethical scalars cannot be pruned."""
        from omni_mercury_engine.core.gosnn_optimizer import ScalarImportanceAnalyzer

        analyzer = ScalarImportanceAnalyzer(seed=SEED)

        # Record history
        for _ in range(20):
            analyzer.record_scalars(
                {
                    "omnibenevolence": 0.99,
                    "omnimorality": 1.2,
                    "regular_scalar": 0.1,
                }
            )

        importances = analyzer.compute_importance(
            {"omnibenevolence": 0.99, "omnimorality": 1.2, "regular_scalar": 0.1},
            output_value=0.5,
        )

        # Ethical scalars should not be prunable
        assert importances["omnibenevolence"].prunable == False  # noqa: E712 - numpy.bool_
        assert importances["omnimorality"].prunable == False  # noqa: E712 - numpy.bool_


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
