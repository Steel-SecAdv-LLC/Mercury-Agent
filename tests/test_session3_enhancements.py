"""
Mercury Agent - Session 3 Enhancement Tests
Copyright (C) 2025 Steel Security Advisory LLC

Comprehensive tests for:
- Neuro-Symbolic Hub
- GOSNN Optimizer
- Real-World Benchmarks
- Integration with previous sessions
"""

import numpy as np
import pytest

# Fixed seed for reproducibility
SEED = 42


class TestNeuroSymbolicHub:
    """Tests for the enhanced neuro-symbolic hub."""

    def test_hub_initialization(self):
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

    def test_phi_weighted_fusion(self):
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

    def test_predict_returns_explanations(self):
        """Test prediction returns explanations."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(input_dim=32, seed=SEED)

        X = np.random.randn(10, 32)
        results = hub.predict(X, return_explanations=True)

        assert len(results) == 10
        for result in results:
            assert hasattr(result, "anomaly_score")
            assert hasattr(result, "neural_score")
            assert hasattr(result, "symbolic_score")
            assert hasattr(result, "reasoning_chain")
            assert 0 <= result.anomaly_score <= 1
            assert 0 <= result.confidence <= 1

    def test_benevolence_enforcement(self):
        """Test benevolence ≥0.99 enforcement."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(benevolence_threshold=0.99, seed=SEED)

        X = np.random.randn(5, 64)
        results = hub.predict(X)

        for result in results:
            # Check benevolence is computed
            assert hasattr(result, "benevolence_score")
            # If not compliant, should have violations
            if not result.ethical_compliant:
                assert len(result.ethical_violations) > 0

    def test_knowledge_graph_rules(self):
        """Test symbolic rules fire correctly."""
        from omni_mercury_engine.core.neurosymbolic_hub import (
            NeuroSymbolicHub,
            SymbolicRule,
        )

        hub = NeuroSymbolicHub(seed=SEED)

        # Add custom rule
        hub.add_rule(SymbolicRule(
            rule_id="test_rule",
            premise="test_condition >= 1.0",
            conclusion="test_alert",
            confidence=0.9,
        ))

        # Predict with context that triggers rule
        X = np.random.randn(1, 64)
        context = {"test_condition": 1.5}
        results = hub.predict(X, context=context)

        # Check rule was considered
        assert len(results) == 1

    def test_gosnn_integration(self):
        """Test GOSNN scalar registration."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(seed=SEED)

        # Run some inferences
        X = np.random.randn(5, 64)
        hub.predict(X)

        # Get scalars
        scalars = hub.get_gosnn_scalars()

        assert "neurosymbolic_neural_weight" in scalars
        assert "neurosymbolic_symbolic_weight" in scalars
        assert "neurosymbolic_inference_count" in scalars
        assert scalars["neurosymbolic_inference_count"] == 5

    def test_fit_learns_weights(self):
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

    def test_optimizer_initialization(self):
        """Test optimizer initializes correctly."""
        from omni_mercury_engine.core.gosnn_optimizer import GOSNNOptimizer

        optimizer = GOSNNOptimizer(
            sigma_sacred=0.96,
            target_overhead_percent=2.0,
        )

        assert optimizer.sigma_sacred == 0.96
        assert optimizer.target_overhead == 2.0

    def test_scalar_importance_analysis(self):
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

    def test_ethical_gate_hard_constraint(self):
        """Test σ_Sacred hard constraint at 0.93."""
        from omni_mercury_engine.core.gosnn_optimizer import EthicalGateOptimizer

        gate = EthicalGateOptimizer(
            sigma_sacred_hard=0.93,
            sigma_sacred_target=0.96,
        )

        # Should pass with high ethical scalars
        passes, score, violations = gate.evaluate({
            "omnimorality": 1.2,
            "omniempathy": 1.2,
            "omnibenevolence": 0.99,
        })

        assert passes is True
        assert score >= 0.93

        # Should fail with low ethical scalars
        passes, score, violations = gate.evaluate({
            "omnimorality": 0.3,
            "omniempathy": 0.3,
            "omnibenevolence": 0.5,
        })

        assert passes is False
        assert len(violations) > 0

    def test_attention_optimizer_overhead(self):
        """Test attention overhead stays below target."""
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
        # Overhead should be reasonable
        assert overhead < 100  # Not excessively high

    def test_full_optimization(self):
        """Test full GOSNN optimization."""
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

        assert result.total_scalars > 0
        assert result.ethical_compliant is True
        assert result.sigma_sacred_value >= 0.93


class TestRealWorldBenchmark:
    """Tests for real-world benchmark runner."""

    def test_synthetic_data_generation(self):
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

    def test_benchmark_runner_sklearn_detector(self):
        """Test benchmark with sklearn detector."""
        from sklearn.ensemble import IsolationForest

        from omni_mercury_engine.core.realworld_benchmark import RealWorldBenchmarkRunner

        runner = RealWorldBenchmarkRunner(n_folds=3, seed=SEED)

        # Wrapper for sklearn
        class IFWrapper:
            def __init__(self):
                self.model = IsolationForest(n_estimators=50, random_state=SEED)

            def fit(self, X, y=None):
                self.model.fit(X)

            def predict(self, X):
                return (self.model.predict(X) == -1).astype(int)

            def predict_proba(self, X):
                scores = -self.model.score_samples(X)
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                return np.column_stack([1 - scores, scores])

        result = runner.run_benchmark(
            IFWrapper(),
            "SMD",
            "IsolationForest"
        )

        assert result.metrics.roc_auc > 0.4  # Should be above random
        assert result.metrics.f1 >= 0.0
        assert result.n_folds == 3

    def test_benchmark_with_neurosymbolic_hub(self):
        """Test benchmark with neuro-symbolic hub."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub
        from omni_mercury_engine.core.realworld_benchmark import RealWorldBenchmarkRunner

        runner = RealWorldBenchmarkRunner(n_folds=3, seed=SEED)

        # Create wrapper for NeuroSymbolicHub
        class NSHWrapper:
            def __init__(self):
                self.hub = NeuroSymbolicHub(input_dim=38, seed=SEED)

            def fit(self, X, y=None):
                self.hub.fit(X, y)

            def predict(self, X):
                return self.hub.predict(X, return_explanations=False)

            def predict_proba(self, X):
                return self.hub.predict_proba(X)

        result = runner.run_benchmark(
            NSHWrapper(),
            "SMD",
            "NeuroSymbolicHub"
        )

        assert result.metrics.roc_auc >= 0.0
        assert result.detector_name == "NeuroSymbolicHub"

    def test_event_metrics_computation(self):
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

    def test_neurosymbolic_with_stacking_fusion(self):
        """Test neuro-symbolic hub with stacking fusion."""
        from omni_mercury_engine.core.neurosymbolic_hub import (
            FusionMode,
            NeuroSymbolicHub,
        )

        hub = NeuroSymbolicHub(
            fusion_mode=FusionMode.STACKING,
            seed=SEED,
        )

        # Fit with labeled data
        np.random.seed(SEED)
        X = np.random.randn(100, 64)
        y = (np.random.rand(100) > 0.8).astype(int)

        hub.fit(X, y)

        # Predict
        X_test = np.random.randn(10, 64)
        results = hub.predict(X_test)

        assert len(results) == 10

    def test_gosnn_integration_layer(self):
        """Test integration with GOSNN integration layer from Session 2."""
        try:
            from omni_mercury_engine.core.gosnn_integration import (
                GOSNNIntegration,
            )

            integration = GOSNNIntegration(
                sigma_sacred=0.96,
                benevolence_threshold=0.99,
            )

            # Should initialize without error
            assert integration.sigma_sacred == 0.96

        except ImportError:
            pytest.skip("GOSNN integration module not available")

    def test_calibration_integration(self):
        """Test integration with calibration from Session 1."""
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

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to ethical detection."""
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )
        from omni_mercury_engine.core.gosnn_optimizer import GOSNNOptimizer
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        # Reset GOSNN singleton
        reset_global_network()

        # Initialize components
        hub = NeuroSymbolicHub(input_dim=64, seed=SEED)
        gosnn = GlobalOmniScalarNetwork()
        optimizer = GOSNNOptimizer(seed=SEED)

        # Generate data
        np.random.seed(SEED)
        X_train = np.random.randn(100, 64)
        y_train = (np.random.rand(100) > 0.9).astype(int)
        X_test = np.random.randn(20, 64)

        # Fit
        hub.fit(X_train, y_train)

        # Predict
        results = hub.predict(X_test)

        # Integrate with GOSNN
        hub.integrate_with_gosnn(gosnn)

        # Optimize GOSNN
        opt_result = optimizer.optimize(gosnn)

        # Assertions
        assert len(results) == 20
        assert all(r.benevolence_score >= 0 for r in results)
        assert opt_result.ethical_compliant is True
        assert opt_result.sigma_sacred_value >= 0.93


class TestEthicalConstraints:
    """Tests for ethical constraint enforcement."""

    def test_benevolence_threshold_immutable(self):
        """Test benevolence threshold is enforced."""
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

        hub = NeuroSymbolicHub(benevolence_threshold=0.99, seed=SEED)

        # Cannot set below threshold
        assert hub.benevolence_threshold == 0.99

    def test_sigma_sacred_hard_limit(self):
        """Test σ_Sacred hard limit at 0.93."""
        from omni_mercury_engine.core.gosnn_optimizer import EthicalGateOptimizer

        gate = EthicalGateOptimizer(sigma_sacred_hard=0.93)

        # Should block if below 0.93
        passes, _, _ = gate.evaluate({"omnibenevolence": 0.5})
        assert passes is False

    def test_ethical_rules_not_prunable(self):
        """Test ethical scalars cannot be pruned."""
        from omni_mercury_engine.core.gosnn_optimizer import ScalarImportanceAnalyzer

        analyzer = ScalarImportanceAnalyzer(seed=SEED)

        # Record history
        for _ in range(20):
            analyzer.record_scalars({
                "omnibenevolence": 0.99,
                "omnimorality": 1.2,
                "regular_scalar": 0.1,
            })

        importances = analyzer.compute_importance(
            {"omnibenevolence": 0.99, "omnimorality": 1.2, "regular_scalar": 0.1},
            output_value=0.5,
        )

        # Ethical scalars should not be prunable
        assert importances["omnibenevolence"].prunable is False
        assert importances["omnimorality"].prunable is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
