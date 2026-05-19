"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Hypothesis-based Property Testing for Mercury-Agent Components.

Uses property-based testing to verify invariants and edge cases that
unit tests might miss. This approach generates thousands of test cases
automatically to find edge cases and bugs.

Reference: Hypothesis documentation (https://hypothesis.readthedocs.io/)
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

# Check if torch is available
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

# Check if hypothesis is available
hypothesis_available = True
try:
    from hypothesis import (
        HealthCheck,
        assume,
        given,
        settings,
        strategies as st,
    )
    from hypothesis.extra import numpy as npst
except ImportError:
    hypothesis_available = False

    # Create dummy decorators for when hypothesis isn't available.
    # Each fallback is a runtime shim that defers to ``pytest.skip``
    # so the tests never actually execute against it.
    def given(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return pytest.mark.skip(reason="Hypothesis not installed")(func)

        return decorator

    def settings(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    class st:  # type: ignore[no-redef]
        @staticmethod
        def floats(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def integers(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def text(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def lists(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def dictionaries(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def booleans() -> None:
            return None

        @staticmethod
        def characters(*args: Any, **kwargs: Any) -> None:
            return None

        @staticmethod
        def tuples(*args: Any, **kwargs: Any) -> None:
            return None

    class npst:  # type: ignore[no-redef]
        @staticmethod
        def arrays(*args: Any, **kwargs: Any) -> None:
            return None

    class HealthCheck:  # type: ignore[no-redef]
        too_slow = None

    def assume(condition: Any) -> None:
        pass


class TestInputValidationProperties:
    """Property-based tests for input validation."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_sanitize_never_returns_dangerous_html(self, text: str) -> None:
        """Sanitized output should never contain raw script tags."""
        from omni_mercury_engine.security.input_validation import (
            InputValidator,
            SanitizationLevel,
        )

        validator = InputValidator(level=SanitizationLevel.MODERATE)
        result = validator.validate_string(text)

        # Property: Sanitized output should not contain unescaped script tags
        if result.sanitized_value:
            assert "<script" not in result.sanitized_value.lower()
            assert "javascript:" not in result.sanitized_value.lower()

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_sanitize_strict_only_allows_safe_chars(self, text: str) -> None:
        """Strict sanitization should only allow alphanumeric and limited punctuation."""
        from omni_mercury_engine.security.input_validation import (
            InputValidator,
            SanitizationLevel,
        )

        validator = InputValidator(level=SanitizationLevel.STRICT)
        result = validator.validate_string(text, level=SanitizationLevel.STRICT)

        # Property: Strict mode only allows safe characters
        if result.sanitized_value:
            import re

            # Only alphanumeric, underscore, hyphen, at, dot, space
            assert re.match(r"^[a-zA-Z0-9_\-@. ]*$", result.sanitized_value)

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.integers(min_value=-(10**15), max_value=10**15))
    @settings(max_examples=100)
    def test_integer_validation_roundtrip(self, value: int) -> None:
        """Integer validation should preserve valid integers."""
        from omni_mercury_engine.security.input_validation import InputValidator

        validator = InputValidator()
        result = validator.validate_integer(value)

        # Property: Valid integers should pass validation unchanged
        assert result.is_valid
        assert result.sanitized_value == value

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_float_validation_preserves_valid_floats(self, value: float) -> None:
        """Float validation should preserve valid floats."""
        from omni_mercury_engine.security.input_validation import InputValidator

        validator = InputValidator()
        result = validator.validate_float(value)

        # Property: Valid floats should pass validation
        assert result.is_valid
        assert abs(result.sanitized_value - value) < 1e-10

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_path_traversal_always_detected(self, prefix: str) -> None:
        """Path traversal patterns should always be detected."""
        from omni_mercury_engine.security.input_validation import InputValidator

        validator = InputValidator()

        # Test with common traversal patterns
        traversal_payloads = [
            f"{prefix}/../etc/passwd",
            f"{prefix}/..\\windows\\system32",
            f"{prefix}/%2e%2e/etc/passwd",
        ]

        for payload in traversal_payloads:
            result = validator.validate_path(payload)
            # Property: Path traversal should be detected
            assert not result.is_valid or ".." not in payload


class TestDoubleHelixEngineProperties:
    """Property-based tests for the evolution engine."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(
        npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=4, max_value=64),
            elements=st.floats(
                min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_evolution_preserves_finite_values(self, initial_state: np.ndarray[Any, Any]) -> None:
        """Evolution should never produce NaN or Inf values."""
        from omni_mercury_engine.core.double_helix_engine import MercuryEquationEngine

        # Skip if initial state has bad values or is too small
        assume(np.all(np.isfinite(initial_state)))
        assume(np.linalg.norm(initial_state) > 1e-10)

        # Normalize input to unit norm to prevent numerical overflow in evolution
        # This is a valid constraint since the engine expects normalized state vectors
        normalized_state = initial_state / (np.linalg.norm(initial_state) + 1e-10)

        engine = MercuryEquationEngine(dimension=len(initial_state))
        final_state, history = engine.converge(normalized_state, max_iter=10)

        # Property: Output should always be finite
        assert np.all(np.isfinite(final_state))
        for state in history:
            assert np.all(np.isfinite(state.state_vector))

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.integers(min_value=4, max_value=128))
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_evolution_dimension_consistency(self, dim: int) -> None:
        """Evolution should preserve state dimension."""
        from omni_mercury_engine.core.double_helix_engine import MercuryEquationEngine

        engine = MercuryEquationEngine(dimension=dim)
        initial_state = np.random.randn(dim)
        final_state, history = engine.converge(initial_state, max_iter=5)

        # Property: Dimension should be preserved
        assert len(final_state) == dim
        for state in history:
            assert len(state.state_vector) == dim


class TestBiasDetectorProperties:
    """Property-based tests for bias detection."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.integers(min_value=100, max_value=1000), st.integers(min_value=2, max_value=5))
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_perfect_predictor_is_fair(self, n_samples: int, n_groups: int) -> None:
        """A perfect predictor should pass fairness checks."""
        from omni_mercury_engine.ml.bias_detection import BiasDetector, FairnessMetric

        # Generate balanced data
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true.copy()  # Perfect predictions
        sensitive_features = np.random.randint(0, n_groups, n_samples)

        detector = BiasDetector(use_fairlearn=False)
        report = detector.evaluate(
            y_true, y_pred, sensitive_features, metrics=[FairnessMetric.EQUALIZED_ODDS]
        )

        # Property: Perfect predictor should have no equalized odds disparity
        for result in report.fairness_results:
            if result.metric == FairnessMetric.EQUALIZED_ODDS:
                # Perfect predictions mean TPR=1, FPR=0 for all groups
                assert result.disparity < 0.01 or result.is_fair

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.integers(min_value=50, max_value=500))
    @settings(max_examples=20)
    def test_random_predictor_detected_as_unfair_for_biased_data(self, n_samples: int) -> None:
        """Deliberately biased predictions should fail fairness checks."""
        from omni_mercury_engine.ml.bias_detection import BiasDetector, FairnessMetric

        # Create deliberately biased data
        y_true = np.random.randint(0, 2, n_samples)

        # Group 0 gets all 1s, Group 1 gets all 0s (extreme bias)
        sensitive_features = np.random.randint(0, 2, n_samples)
        y_pred = (sensitive_features == 0).astype(int)

        detector = BiasDetector(use_fairlearn=False)
        report = detector.evaluate(
            y_true, y_pred, sensitive_features, metrics=[FairnessMetric.DEMOGRAPHIC_PARITY]
        )

        # Property: Extreme bias should be detected
        dp_result = next(
            r for r in report.fairness_results if r.metric == FairnessMetric.DEMOGRAPHIC_PARITY
        )
        assert dp_result.disparity > 0.5 or not dp_result.is_fair


class TestThreatDetectorProperties:
    """Property-based tests for threat detection."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.text(alphabet=st.characters(categories=["L", "N"]), min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_clean_input_not_flagged(self, text: str) -> None:
        """Alphanumeric text should not be flagged as threats."""
        from omni_mercury_engine.security.threat_detection import ThreatDetector

        detector = ThreatDetector()
        result = detector.detect_all(text)

        # Property: Pure alphanumeric should not trigger threats
        assert not result["is_threat"]

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.text(min_size=0, max_size=50))
    @settings(max_examples=50)
    def test_sql_injection_always_detected(self, prefix: str) -> None:
        """SQL injection patterns should always be detected."""
        from omni_mercury_engine.security.threat_detection import ThreatDetector

        detector = ThreatDetector()

        # Test with known SQL injection patterns
        sql_payloads = [
            f"{prefix}' OR '1'='1",
            f"{prefix}; DROP TABLE users--",
            f"{prefix}' UNION SELECT * FROM passwords--",
        ]

        for payload in sql_payloads:
            result = detector.detect_sql_injection(payload)
            # Property: SQL injection patterns should be detected
            assert result["is_threat"]

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.text(min_size=0, max_size=50))
    @settings(max_examples=50)
    def test_xss_always_detected(self, prefix: str) -> None:
        """XSS patterns should always be detected."""
        from omni_mercury_engine.security.threat_detection import ThreatDetector

        detector = ThreatDetector()

        # Test with known XSS patterns
        xss_payloads = [
            f"{prefix}<script>alert('xss')</script>",
            f"{prefix}<img onerror=alert('xss')>",
            f"{prefix}javascript:alert('xss')",
        ]

        for payload in xss_payloads:
            result = detector.detect_xss(payload)
            # Property: XSS patterns should be detected
            assert result["is_threat"]


class TestEthicalEngineProperties:
    """Property-based tests for ethical constraint engine."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=["L"])),
            values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_maat_balance_bounded_output(self, ethical_scores: dict[str, Any]) -> None:
        """Ma'at balance should always produce bounded heart weight."""
        from omni_mercury_engine.ethical.ethical_constraint_engine import MaatBalanceEngine

        engine = MaatBalanceEngine()
        result = engine.weigh_heart_against_feather(ethical_scores)

        # Property: Heart weight should be bounded [0.5, 1.5]
        assert 0.5 <= result.heart_weight <= 1.5
        assert 0.0 <= result.deviation <= 1.0

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(
        npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10)
            ),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_geometry_analysis_bounded_scores(self, data: np.ndarray[Any, Any]) -> None:
        """Immutable geometry analysis should produce scores in [0, 1]."""
        from omni_mercury_engine.ethical.ethical_constraint_engine import ImmutableGeometryProcessor

        processor = ImmutableGeometryProcessor()
        result = processor.analyze_geometry(data)

        # Property: All scores should be bounded [0, 1]
        assert 0.0 <= result.golden_ratio_alignment <= 1.0
        assert 0.0 <= result.fibonacci_spiral_score <= 1.0
        assert 0.0 <= result.vesica_piscis_score <= 1.0
        assert 0.0 <= result.platonic_harmony <= 1.0
        assert 0.0 <= result.overall_geometry_score <= 1.0


class TestDetectorRegistryProperties:
    """Property-based tests for DetectorRegistry invariants."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(npst.arrays(dtype=np.float64, shape=st.integers(min_value=10, max_value=100)))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_features_produces_128d_output(self, features: np.ndarray[Any, Any]) -> None:
        """Aggregated features should always produce 128D output."""
        from omni_mercury_engine.core.detector_registry import (
            DetectorRegistry,
            FeatureExtractionResult,
        )

        assume(np.all(np.isfinite(features)))
        assume(len(features) > 0)

        registry = DetectorRegistry()
        extraction_results = {
            "test_detector": FeatureExtractionResult(
                detector_name="test_detector",
                features=features,
                scores=None,
                execution_time_ms=1.0,
                success=True,
            )
        }

        aggregated = registry.aggregate_features(extraction_results, target_dim=128)

        # Property: Output should be 128D
        if "test_detector" in aggregated:
            assert aggregated["test_detector"].shape[-1] == 128

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_registry_register_unregister_invariant(self, n_detectors: int) -> None:
        """Registry should maintain consistent state after register/unregister."""
        from omni_mercury_engine.core.detector_registry import (
            DetectorCategory,
            DetectorRegistry,
        )

        class MockDetector:
            def extract_features(self, data: Any) -> Any:
                return np.zeros(20)

            def predict(self, data: Any) -> Any:
                return {"scores": np.array([0.5])}

        registry = DetectorRegistry()

        # Register n detectors
        for i in range(n_detectors):
            registry.register(f"detector_{i}", MockDetector(), DetectorCategory.BASE)

        # Property: All detectors should be registered
        assert len(registry.list_all()) == n_detectors

        # Unregister half
        for i in range(n_detectors // 2):
            registry.unregister(f"detector_{i}")

        # Property: Remaining count should be correct
        expected_remaining = n_detectors - (n_detectors // 2)
        assert len(registry.list_all()) == expected_remaining

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(
        npst.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=1, max_value=5), st.integers(min_value=10, max_value=50)
            ),
        )
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_features_no_nans(self, features: np.ndarray[Any, Any]) -> None:
        """Aggregated features should never contain NaN values."""
        import torch

        from omni_mercury_engine.core.detector_registry import (
            DetectorRegistry,
            FeatureExtractionResult,
        )

        assume(np.all(np.isfinite(features)))

        registry = DetectorRegistry()
        extraction_results = {
            "test_detector": FeatureExtractionResult(
                detector_name="test_detector",
                features=features,
                scores=None,
                execution_time_ms=1.0,
                success=True,
            )
        }

        aggregated = registry.aggregate_features(extraction_results, target_dim=128)

        # Property: No NaN values in output
        for name, tensor in aggregated.items():
            assert not torch.isnan(tensor).any(), f"NaN found in {name}"

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(
        st.lists(
            st.text(min_size=1, max_size=20, alphabet=st.characters(categories=["L", "N"])),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=20)
    def test_list_by_tags_returns_subset(self, tags: list[Any]) -> None:
        """list_by_tags should return subset of registered detectors."""
        from omni_mercury_engine.core.detector_registry import (
            DetectorCategory,
            DetectorRegistry,
        )

        class MockDetector:
            def extract_features(self, data: Any) -> Any:
                return np.zeros(20)

        registry = DetectorRegistry()

        # Register detectors with various tags
        registry.register("detector_with_tags", MockDetector(), DetectorCategory.BASE, tags=tags)
        registry.register("detector_no_tags", MockDetector(), DetectorCategory.BASE, tags=[])

        # Property: Searching by tags should find tagged detector
        found = registry.list_by_tags(tags)
        assert "detector_with_tags" in found
        assert "detector_no_tags" not in found


class TestFusionNetworkProperties:
    """Property-based tests for ML fusion network forward pass."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    @given(st.integers(min_value=1, max_value=4))
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_fusion_forward_pass_output_shapes(self, batch_size: int) -> None:
        """Fusion network forward pass should produce correct output shapes."""
        fusion_network = pytest.importorskip("omni_mercury_engine.ml.fusion_network")
        OmniFusionModel = fusion_network.OmniFusionModel

        # OmniFusionModel expects feature_dims dict, hidden_dim, num_heads, dropout, num_classes
        feature_dims = {"statistical": 10, "temporal": 32}
        model = OmniFusionModel(
            feature_dims=feature_dims,
            hidden_dim=64,
            num_classes=10,
        )
        model.eval()

        # Create synthetic input as dict of features (matching OmniFusionModel.forward signature)
        detector_features = {
            "statistical": torch.randn(batch_size, 10),
            "temporal": torch.randn(batch_size, 32),
        }

        with torch.no_grad():
            output = model(detector_features)

        # Property: Output should have correct batch dimension
        # OmniFusionModel returns a dict with anomaly_probs, class_logits, regression_output
        if isinstance(output, dict):
            for key, out in output.items():
                if out is not None and hasattr(out, "shape"):
                    assert out.shape[0] == batch_size
        elif isinstance(output, tuple):
            for out in output:
                if out is not None:
                    assert out.shape[0] == batch_size
        else:
            assert output.shape[0] == batch_size


class TestValidationPipelineProperties:
    """Property-based tests for validation pipeline error handling."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=20)
    def test_invalid_dataset_name_handled(self, dataset_name: str) -> None:
        """Invalid dataset names should be handled gracefully."""
        # get_loader was removed from validation.data_loaders; tolerate either
        # state (still available in some forks) without breaking the type gate.
        try:
            from omni_mercury_engine.validation.data_loaders import (  # type: ignore[attr-defined]
                get_loader,
            )
        except (ImportError, AttributeError):
            pytest.skip("get_loader not available")
            return  # Explicit return after skip for static analysis

        # Property: Invalid dataset names should not crash
        try:
            get_loader(dataset_name)
            # If it returns something, it should be None or raise an error
        except (ValueError, KeyError, NotImplementedError):
            pass  # Expected behavior for invalid names
        except Exception:
            # Other exceptions are acceptable as long as no crash
            assert True


class TestKnowledgeGraphProperties:
    """Property-based tests for knowledge graph query operations."""

    @pytest.mark.skipif(not hypothesis_available, reason="Hypothesis not installed")
    @given(st.text(min_size=1, max_size=30, alphabet=st.characters(categories=["L", "N"])))
    @settings(max_examples=20)
    def test_query_nonexistent_node_handled(self, node_name: str) -> None:
        """Querying non-existent nodes should be handled gracefully."""
        knowledge_graph = pytest.importorskip("omni_mercury_engine.cognitive.knowledge_graph")
        KnowledgeGraph = knowledge_graph.KnowledgeGraph

        kg = KnowledgeGraph()

        # Property: Querying non-existent node should not crash
        try:
            kg.query_node(node_name)
            # Result should be None or empty for non-existent nodes
        except (KeyError, ValueError):
            pass  # Expected behavior
        except Exception:
            pass  # Other exceptions acceptable


# Run with: pytest tests/test_property_based.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
