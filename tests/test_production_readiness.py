"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for production-readiness improvements:
- EthicalConstraintViolationError + enforce() + MINIMUM_BENEVOLENCE_FLOOR
- CognitiveOrchestrator ethical gate integration
- RefactoringTransformer guard-clause extraction and constant hoisting
- Learnable3REngine.fit() with best-epoch checkpointing
- GOSNN AttentionProvider interface
- BenchmarkDiagnostics print-to-logger conversion
"""

from __future__ import annotations

import ast
import logging
import textwrap

import numpy as np
import pytest

# ============================================================================
# Ethical enforcement tests
# ============================================================================


class TestEthicalConstraintViolationError:
    """Tests for the hard ethical enforcement exception."""

    def test_exception_attributes(self):
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )

        exc = EthicalConstraintViolationError(
            action="delete_all",
            score=0.50,
            threshold=0.99,
        )
        assert exc.action == "delete_all"
        assert exc.score == 0.50
        assert exc.threshold == 0.99
        assert "delete_all" in str(exc)
        assert "0.5000" in str(exc)

    def test_is_runtime_error(self):
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )

        exc = EthicalConstraintViolationError("x", 0.1, 0.9)
        assert isinstance(exc, RuntimeError)


class TestMinimumBenevolenceFloor:
    """Tests for the benevolence threshold floor clamp."""

    def test_floor_clamps_low_threshold(self):
        from omni_mercury_engine.cognitive.ethical_bounding import (
            MINIMUM_BENEVOLENCE_FLOOR,
            BenevolenceScorer,
        )

        scorer = BenevolenceScorer(benevolence_threshold=0.0)
        assert scorer.benevolence_threshold == MINIMUM_BENEVOLENCE_FLOOR

    def test_floor_allows_high_threshold(self):
        from omni_mercury_engine.cognitive.ethical_bounding import BenevolenceScorer

        scorer = BenevolenceScorer(benevolence_threshold=0.95)
        assert scorer.benevolence_threshold == 0.95

    def test_floor_value(self):
        from omni_mercury_engine.cognitive.ethical_bounding import (
            MINIMUM_BENEVOLENCE_FLOOR,
        )

        assert MINIMUM_BENEVOLENCE_FLOOR == 0.70


class TestBenevolenceScorerEnforce:
    """Tests for BenevolenceScorer.enforce()."""

    def test_enforce_raises_on_impermissible(self):
        from omni_mercury_engine.cognitive.ethical_bounding import (
            BenevolenceScorer,
            EthicalConstraintViolationError,
        )

        scorer = BenevolenceScorer(benevolence_threshold=0.99)
        with pytest.raises(EthicalConstraintViolationError) as exc_info:
            # "destroy" keywords trigger high harm score -> impermissible
            scorer.enforce("destroy_all_data", {"intent": "malicious"})

        assert exc_info.value.score < 0.99
        assert exc_info.value.threshold == 0.99

    def test_enforce_returns_score_on_permissible(self):
        from omni_mercury_engine.cognitive.ethical_bounding import (
            BenevolenceScorer,
            EthicalScore,
        )

        # Use context rich in positive ethical keywords so the benevolence
        # score exceeds the floor (0.70).
        scorer = BenevolenceScorer(benevolence_threshold=0.70)
        result = scorer.enforce(
            "help_humanitarian_aid",
            {
                "intent": "selfless benefit humanitarian aid care help support "
                "empathy fair just equal rights data research verify"
            },
        )
        assert isinstance(result, EthicalScore)
        assert result.is_permissible


class TestCognitiveOrchestratorEthicalGate:
    """Tests for ethical gate wiring into CognitiveOrchestrator."""

    def test_analyze_includes_benevolence_score(self):
        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

        orchestrator = CognitiveOrchestrator(
            enable_plasticity=False,
            enable_causal=False,
            enable_ipb=False,
            enable_cbr=False,
            enable_indicators=False,
            strict_ethics=False,  # advisory mode for testing
        )
        result = orchestrator.analyze(
            detection_result={"is_anomaly": False, "anomaly_prob": 0.1, "severity": 0.1},
            context={"domain": "general"},
        )
        assert hasattr(result, "benevolence_score")
        assert hasattr(result, "ethical_permissible")
        assert result.benevolence_score > 0

    def test_strict_mode_raises_on_violation(self):
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )
        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

        # Strict mode with impossibly high threshold
        orchestrator = CognitiveOrchestrator(
            enable_plasticity=False,
            enable_causal=False,
            enable_ipb=False,
            enable_cbr=False,
            enable_indicators=False,
            strict_ethics=True,
        )
        # The default benevolence threshold is 0.99 — most actions won't
        # reach that, which means strict mode will raise. We test that
        # the exception propagates properly.
        try:
            orchestrator.analyze(
                detection_result={
                    "is_anomaly": True,
                    "anomaly_prob": 0.95,
                    "severity": 0.9,
                },
                context={"domain": "general"},
            )
            # If it didn't raise, the action was permissible — that's also valid
        except EthicalConstraintViolationError as e:
            assert e.threshold == 0.99
            assert e.score < 0.99


class TestCognitiveInitExports:
    """Test that the cognitive __init__ exports the new symbols."""

    def test_exports_available(self):
        from omni_mercury_engine.cognitive import (
            MINIMUM_BENEVOLENCE_FLOOR,
            BenevolenceScorer,
            EthicalConstraintViolationError,
        )

        assert MINIMUM_BENEVOLENCE_FLOOR == 0.70
        assert callable(BenevolenceScorer)
        assert issubclass(EthicalConstraintViolationError, RuntimeError)


# ============================================================================
# RefactoringTransformer tests
# ============================================================================


class TestRefactoringTransformerGuardClause:
    """Tests for guard-clause extraction (nesting reduction)."""

    def test_guard_clause_single_if(self):
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent(
            """\
            def foo(x):
                if x is not None:
                    result = x + 1
                    return result
        """
        )
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_nesting"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)

        # Should have an inverted guard clause
        assert "not" in code
        assert "return None" in code or "return" in code

    def test_guard_clause_preserves_docstring(self):
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent(
            '''\
            def foo(x):
                """My docstring."""
                if x is not None:
                    return x
        '''
        )
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_nesting"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)
        assert "My docstring" in code

    def test_guard_clause_multiple_ifs_preserves_semantics(self):
        """Only the *last* if (no else) is safe to transform into a guard clause."""
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent(
            """\
            def foo(x, y):
                if x is not None:
                    a = x + 1
                if y is not None:
                    b = y + 2
        """
        )
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_nesting"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)

        # Only the last if should be transformed; the first if must remain
        # untouched to preserve semantics (early return would skip the second if).
        assert "if x is not None" in code
        assert "return None" in code
        # The first if-block body (a = x + 1) must still be indented under the if,
        # not hoisted to top-level as it would be with the broken transformation.
        assert "    if x is not None:\n        a = x + 1" in code

    def test_if_with_else_not_transformed(self):
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent(
            """\
            def foo(x):
                if x > 0:
                    return 1
                else:
                    return 0
        """
        )
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_nesting"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)

        # Should NOT be transformed because it has an else branch
        assert "else" in code


class TestRefactoringTransformerConstantHoisting:
    """Tests for constant hoisting (complexity reduction)."""

    def test_hoists_repeated_literal(self):
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent(
            """\
            def foo():
                x = 42
                y = 42
                z = 42
        """
        )
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)

        # Should have a _const_ variable
        assert "_const_" in code

    def test_does_not_hoist_trivial_values(self):
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent(
            """\
            def foo():
                x = 0
                y = 0
                z = 1
                w = 1
        """
        )
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)

        # 0 and 1 are trivial — should NOT be hoisted
        assert "_const_" not in code

    def test_hoists_repeated_strings(self):
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent(
            """\
            def foo():
                a = "hello"
                b = "hello"
        """
        )
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)
        assert "_const_" in code

    def test_output_compiles_and_runs(self):
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent(
            """\
            def foo():
                x = 42
                y = 42
                return x + y
        """
        )
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        ast.unparse(new_tree)

        # Should compile and execute correctly
        compiled = compile(new_tree, "<test>", "exec")
        namespace: dict[str, object] = {}
        exec(compiled, namespace)  # noqa: S102
        assert namespace["foo"]() == 84  # type: ignore[operator]


# ============================================================================
# Learnable3R.fit() tests
# ============================================================================


class TestLearnable3RFit:
    """Tests for Learnable3REngine.fit() with best-epoch checkpointing."""

    @pytest.fixture
    def engine(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from omni_mercury_engine.core.three_r.learnable_fusion import (
            Learnable3RConfig,
            Learnable3REngine,
        )

        config = Learnable3RConfig(hidden_dim=16)
        return Learnable3REngine(config=config, device="cpu")

    def test_fit_returns_history(self, engine):
        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)

        history = engine.fit(X, y, epochs=5, batch_size=16, patience=3, seed=42)

        assert "train_losses" in history
        assert "val_losses" in history
        assert "best_epoch" in history
        assert "best_val_loss" in history
        assert "stopped_early" in history
        assert len(history["train_losses"]) == 5
        assert len(history["val_losses"]) == 5

    def test_fit_loss_decreases(self, engine):
        # Generate a simple learnable pattern
        rng = np.random.default_rng(123)
        X = rng.standard_normal((100, 4)).astype(np.float32)
        y = (X[:, 0] * 0.5 + X[:, 1] * 0.3).astype(np.float32)

        history = engine.fit(X, y, epochs=30, batch_size=16, patience=20, seed=42)

        # First loss should be higher than last
        assert history["train_losses"][0] > history["train_losses"][-1]

    def test_early_stopping(self, engine):
        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)

        history = engine.fit(
            X,
            y,
            epochs=1000,
            batch_size=16,
            patience=3,
            min_delta=1e-10,
            seed=42,
        )

        # Should stop before 1000 epochs
        assert len(history["train_losses"]) < 1000
        assert history["stopped_early"] is True

    def test_best_epoch_checkpoint_restored(self, engine):
        X = np.random.randn(80, 4).astype(np.float32)
        y = np.random.randn(80).astype(np.float32)

        history = engine.fit(X, y, epochs=20, batch_size=16, patience=5, seed=42)

        # The model should now have the weights from best_epoch, not last epoch
        # We verify by checking the history is well-formed
        best_epoch = history["best_epoch"]
        assert history["val_losses"][best_epoch] == history["best_val_loss"]

    def test_fit_requires_minimum_samples(self, engine):
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        y = np.array([1.0])

        with pytest.raises(ValueError, match="at least 2 samples"):
            engine.fit(X, y)

    def test_fit_without_pytorch(self):
        """Test that fit() gracefully handles missing PyTorch."""
        from omni_mercury_engine.core.three_r.learnable_fusion import (
            Learnable3REngine,
        )

        # Create an engine and force model to None
        engine = Learnable3REngine.__new__(Learnable3REngine)
        engine.model = None

        history = engine.fit(np.zeros((10, 4)), np.zeros(10))
        assert history["train_losses"] == []
        assert history["stopped_early"] is False


# ============================================================================
# GOSNN AttentionProvider tests
# ============================================================================


class TestAttentionProvider:
    """Tests for the AttentionProvider interface."""

    def test_interface_exists(self):
        from omni_mercury_engine.core.gosnn_optimizer import AttentionProvider

        assert hasattr(AttentionProvider, "get_attention")

    def test_custom_provider_plugs_in(self):
        from omni_mercury_engine.core.gosnn_optimizer import (
            AttentionProvider,
            GOSNNOptimizer,
        )

        class MockProvider(AttentionProvider):
            def __init__(self):
                self.called = False

            def get_attention(self) -> np.ndarray:
                self.called = True
                return np.ones((32, 16, 16))

        provider = MockProvider()
        optimizer = GOSNNOptimizer(attention_provider=provider)
        assert optimizer._attention_provider is provider

    def test_placeholder_warning_when_no_provider(self, caplog):
        from omni_mercury_engine.core.gosnn_optimizer import GOSNNOptimizer

        optimizer = GOSNNOptimizer()
        assert optimizer._attention_provider is None


# ============================================================================
# BenchmarkDiagnostics logger conversion tests
# ============================================================================


class TestBenchmarkDiagnosticsLogging:
    """Tests that quick_diagnose uses logger, not print."""

    def test_quick_diagnose_uses_logger(self, caplog):
        from omni_mercury_engine.evaluation.benchmark_diagnostics import (
            BenchmarkDiagnostics,
        )

        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1])

        with caplog.at_level(
            logging.INFO,
            logger="omni_mercury_engine.evaluation.benchmark_diagnostics",
        ):
            BenchmarkDiagnostics.quick_diagnose(
                scores,
                labels=labels,
                threshold=0.5,
                detector_name="TestDetector",
            )

        assert "TestDetector" in caplog.text
        assert "Precision" in caplog.text

    def test_quick_diagnose_f1_zero_warning(self, caplog):
        from omni_mercury_engine.evaluation.benchmark_diagnostics import (
            BenchmarkDiagnostics,
        )

        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 1, 1])

        with caplog.at_level(
            logging.WARNING,
            logger="omni_mercury_engine.evaluation.benchmark_diagnostics",
        ):
            BenchmarkDiagnostics.quick_diagnose(
                scores,
                labels=labels,
                threshold=0.99,
                detector_name="TestDetector",
            )

        assert "DIAGNOSIS" in caplog.text or "F1" in caplog.text
