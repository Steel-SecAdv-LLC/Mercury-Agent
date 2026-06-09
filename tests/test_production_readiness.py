# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive tests for production-readiness improvements: - EthicalConstraintViolationError + enforce() + MINIMUM_BENEVOLENCE_FLOOR - CognitiveOrchestrator ethical gate integration - RefactoringTransformer guard-clause extraction and constant hoisting - Learnable3REngine.fit() with best-epoch checkpointing - GOSNN AttentionProvider interface - BenchmarkDiagnostics print-to-logger conversion."""

from __future__ import annotations

import ast
import logging
import textwrap
from typing import Any

import numpy as np
import pytest

# Note: tests/conftest.py installs an autouse
# ``_restore_engine_logger_propagation`` fixture that resets
# ``logging.getLogger("omni_mercury_engine").propagate`` between tests so
# the caplog-based assertions below remain reliable even if another test
# class exercises ``configure_logging`` (which flips propagate to False).

# ============================================================================
# Ethical enforcement tests
# ============================================================================


class TestEthicalConstraintViolationError:
    """Tests for the hard ethical enforcement exception."""

    def test_exception_attributes(self) -> None:
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

    def test_is_runtime_error(self) -> None:
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )

        exc = EthicalConstraintViolationError("x", 0.1, 0.9)
        assert isinstance(exc, RuntimeError)


class TestMinimumBenevolenceFloor:
    """Tests for the benevolence threshold floor clamp."""

    def test_floor_clamps_low_threshold(self) -> None:
        from omni_mercury_engine.cognitive.ethical_bounding import (
            MINIMUM_BENEVOLENCE_FLOOR,
            BenevolenceScorer,
        )

        scorer = BenevolenceScorer(benevolence_threshold=0.0)
        assert scorer.benevolence_threshold == MINIMUM_BENEVOLENCE_FLOOR

    def test_floor_allows_high_threshold(self) -> None:
        from omni_mercury_engine.cognitive.ethical_bounding import BenevolenceScorer

        scorer = BenevolenceScorer(benevolence_threshold=0.95)
        assert scorer.benevolence_threshold == 0.95

    def test_floor_value(self) -> None:
        from omni_mercury_engine.cognitive.ethical_bounding import (
            MINIMUM_BENEVOLENCE_FLOOR,
        )

        assert MINIMUM_BENEVOLENCE_FLOOR == 0.70


class TestBenevolenceScorerEnforce:
    """Tests for BenevolenceScorer.enforce()."""

    def test_enforce_raises_on_impermissible(self) -> None:
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

    def test_enforce_returns_score_on_permissible(self) -> None:
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
    """Tests for ethical gate wiring into CognitiveOrchestrator.

    The σ_Immutable gate fails closed with
    ``EthicalConstraintViolationError(check='gosnn_unavailable')`` when
    torch is absent (the documented Wave B contract in
    ``tests/ethical/test_hard_enforcement.py``).  Exercising
    ``orchestrator.analyze`` therefore requires torch — without it the
    gate would reject every call, which is the correct safeguard
    behaviour but not what these benevolence-wiring tests are designed
    to verify.  Skip cleanly at the class boundary.
    """

    def test_analyze_includes_benevolence_score(self) -> None:
        """analyze() always scores benevolence; the result exposes both
        the score and the permissibility flag."""
        pytest.importorskip("torch")
        import warnings

        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

        # ``strict_ethics=False`` is deprecated and ignored — passing it
        # must not silently disable the gate, only emit a warning.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            orchestrator = CognitiveOrchestrator(
                enable_plasticity=False,
                enable_causal=False,
                enable_ipb=False,
                enable_cbr=False,
                enable_indicators=False,
                strict_ethics=False,
            )
            assert orchestrator.strict_ethics is True
            assert any(
                issubclass(w.category, DeprecationWarning)
                and "strict_ethics=False" in str(w.message)
                for w in captured
            )

        result = orchestrator.analyze(
            detection_result={"is_anomaly": False, "anomaly_prob": 0.1, "severity": 0.1},
            context={"domain": "general"},
        )
        assert hasattr(result, "benevolence_score")
        assert hasattr(result, "ethical_permissible")
        assert result.benevolence_score > 0

    def test_strict_mode_raises_on_violation(self) -> None:
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )
        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

        orchestrator = CognitiveOrchestrator(
            enable_plasticity=False,
            enable_causal=False,
            enable_ipb=False,
            enable_cbr=False,
            enable_indicators=False,
            strict_ethics=True,
        )
        # The orchestrator's internal scorer is initialized with
        # MINIMUM_BENEVOLENCE_FLOOR (0.70), and analyze() injects
        # positive-keyword text that comfortably exceeds that floor.
        # To deterministically exercise the violation path, pin the
        # scorer's threshold above the maximum achievable score.
        orchestrator._benevolence_scorer.benevolence_threshold = 1.01

        with pytest.raises(EthicalConstraintViolationError) as exc_info:
            orchestrator.analyze(
                detection_result={
                    "is_anomaly": True,
                    "anomaly_prob": 0.95,
                    "severity": 0.9,
                },
                context={"domain": "general"},
            )
        assert exc_info.value.threshold == 1.01
        assert exc_info.value.score < 1.01


class TestCognitiveInitExports:
    """Test that the cognitive __init__ exports the new symbols."""

    def test_exports_available(self) -> None:
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

    def test_guard_clause_single_if(self) -> None:
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo(x):
                if x is not None:
                    result = x + 1
                    return result
        """)
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_nesting"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Walk the AST to verify the guard-clause rewrite *structurally*.
        # A ``not`` substring would be present in the ORIGINAL source
        # already (``is not None``), and a ``return`` substring is
        # produced by the original ``return result`` line, so a textual
        # match could not distinguish a real rewrite from a no-op.
        func_def = new_tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)

        # Contract of ``_reduce_nesting`` for this input:
        #   1. The function body's first statement must be the inverted
        #      guard ``if not (x is not None): return None`` — i.e. an
        #      ``ast.If`` whose test is an ``ast.UnaryOp(op=ast.Not, ...)``
        #      and whose body is exactly one ``ast.Return``.
        #   2. The original body statements (``result = x + 1`` and
        #      ``return result``) must now be SIBLINGS of that guard at
        #      function-body scope, NOT nested inside any ``ast.If``.
        # Either contract failing means we accepted a no-op silently.
        first = func_def.body[0]
        assert isinstance(
            first, ast.If
        ), f"expected guard clause as first statement, got {type(first).__name__}"
        assert isinstance(first.test, ast.UnaryOp) and isinstance(
            first.test.op, ast.Not
        ), "guard test must be an inverted predicate (UnaryOp(Not))"
        assert (
            len(first.body) == 1 and isinstance(first.body[0], ast.Return) and not first.orelse
        ), "guard body must be a single early-return with no else branch"

        # The former nested body must now live at function-body scope.
        rest = func_def.body[1:]
        assert len(rest) == 2, f"expected 2 hoisted statements, got {len(rest)}"
        assert isinstance(
            rest[0], ast.Assign
        ), f"first hoisted stmt must be the original assignment, got {type(rest[0]).__name__}"
        assert isinstance(
            rest[1], ast.Return
        ), f"second hoisted stmt must be the original return, got {type(rest[1]).__name__}"

        # And the rewritten function must still execute correctly.
        compiled = compile(new_tree, "<test-guard>", "exec")
        namespace: dict[str, object] = {}
        exec(compiled, namespace)  # noqa: S102
        assert namespace["foo"](5) == 6  # type: ignore[operator]
        assert namespace["foo"](None) is None  # type: ignore[operator]

    def test_guard_clause_preserves_docstring(self) -> None:
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent('''\
            def foo(x):
                """My docstring."""
                if x is not None:
                    return x
        ''')
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_nesting"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)
        assert "My docstring" in code

    def test_guard_clause_multiple_ifs_preserves_semantics(self) -> None:
        """Only the *last* if (no else) is safe to transform into a guard clause."""
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo(x, y):
                if x is not None:
                    a = x + 1
                if y is not None:
                    b = y + 2
        """)
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_nesting"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Walk the AST to verify the structural contract directly,
        # rather than asserting on whitespace in ``ast.unparse()`` output
        # (which can change across Python patch releases).
        func_def = new_tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)

        # Contract for two-sibling-ifs input:
        #   * The FIRST ``if x is not None:`` must remain an ``ast.If``
        #     at function-body scope with its original ``Assign`` still
        #     in ``body``.  Hoisting it would let the early-return guard
        #     skip the second ``if`` (semantics-breaking).
        #   * The transformer is allowed to convert the LAST ``if`` into
        #     a guard clause; that's the only safe rewrite here.
        first_if = func_def.body[0]
        assert isinstance(first_if, ast.If), (
            "first statement must remain an ast.If — hoisting it would "
            "let the guard's early return skip the second if-block."
        )
        # Its test must still be the original predicate, not an
        # inverted ``not (x is not None)`` guard.
        assert not (
            isinstance(first_if.test, ast.UnaryOp) and isinstance(first_if.test.op, ast.Not)
        ), "first if's predicate was inverted — that is the broken transform."
        # And the original assignment must still live inside its body.
        assert len(first_if.body) == 1 and isinstance(
            first_if.body[0], ast.Assign
        ), "first if's body must still contain the original assignment."

        # Confirm the function still compiles and runs cleanly with both
        # branches reachable.
        compiled = compile(new_tree, "<test-guard-multi>", "exec")
        namespace: dict[str, object] = {}
        exec(compiled, namespace)  # noqa: S102
        # No ``return`` in source, so ``foo`` returns ``None`` either way;
        # the important contract is that no exception is raised and both
        # branches' assignments execute when their predicates hold.
        assert namespace["foo"](1, 2) is None  # type: ignore[operator]
        assert namespace["foo"](None, 2) is None  # type: ignore[operator]
        assert namespace["foo"](1, None) is None  # type: ignore[operator]

    def test_if_with_else_not_transformed(self) -> None:
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo(x):
                if x > 0:
                    return 1
                else:
                    return 0
        """)
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

    def test_hoists_repeated_literal(self) -> None:
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo():
                x = 42
                y = 42
                z = 42
        """)
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)

        # Should have a _const_ variable
        assert "_const_" in code

    def test_does_not_hoist_trivial_values(self) -> None:
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo():
                x = 0
                y = 0
                z = 1
                w = 1
        """)
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)

        # 0 and 1 are trivial — should NOT be hoisted
        assert "_const_" not in code

    def test_hoists_repeated_strings(self) -> None:
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo():
                a = "hello"
                b = "hello"
        """)
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        code = ast.unparse(new_tree)
        assert "_const_" in code

    def test_output_compiles_and_runs(self) -> None:
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo():
                x = 42
                y = 42
                return x + y
        """)
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

    def test_does_not_hoist_default_arguments(self) -> None:
        """Regression: hoisting must not touch args.defaults / decorator_list.

        Constants in default-argument positions are evaluated at function-
        definition time in the enclosing scope. Replacing them with
        ``_const_N`` references — when the assignment lives inside the
        function body — would raise NameError at definition time.
        """
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo(x=42):
                y = 42
                return x + y
        """)
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Default argument must remain a literal, not a Name reference.
        func_def = new_tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert len(func_def.args.defaults) == 1
        assert isinstance(func_def.args.defaults[0], ast.Constant)
        assert func_def.args.defaults[0].value == 42

        # The function must compile and execute without NameError.
        compiled = compile(new_tree, "<test>", "exec")
        namespace: dict[str, object] = {}
        exec(compiled, namespace)  # noqa: S102
        assert namespace["foo"]() == 84  # type: ignore[operator]
        assert namespace["foo"](100) == 142  # type: ignore[operator]

    def test_does_not_rewrite_inside_fstring(self) -> None:
        """Regression: ``ast.JoinedStr.values`` must contain only ``Constant``/
        ``FormattedValue`` children — replacing a ``Constant`` direct child
        with an ``ast.Name`` produces an AST that ``compile()`` rejects.
        The hoister must skip f-strings entirely so the output always
        compiles cleanly.
        """
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def foo():
                a = f"prefix-greeting"
                b = f"suffix-greeting"
                c = "greeting"
                d = "greeting"
                return (a, b, c, d)
        """)
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Output MUST still compile.
        compiled = compile(new_tree, "<test-fstring>", "exec")
        namespace: dict[str, object] = {}
        exec(compiled, namespace)  # noqa: S102
        assert namespace["foo"]() == (  # type: ignore[operator]
            "prefix-greeting",
            "suffix-greeting",
            "greeting",
            "greeting",
        )

    def test_does_not_rewrite_inside_match_pattern(self) -> None:
        """Regression: ``MatchValue.value`` must remain literal-bearing —
        replacing it with an ``ast.Name`` either fails to compile or
        silently turns the case arm into a ``MatchAs`` capture pattern
        that matches anything.  The hoister must skip ``ast.Match``
        entirely.
        """
        from omni_mercury_engine.core.three_r_mechanism import RefactoringTransformer

        source = textwrap.dedent("""\
            def classify(x):
                a = 42
                b = 42
                match x:
                    case 42:
                        return ("answer", a + b)
                    case 100:
                        return ("century", a + b)
                    case _:
                        return ("other", a + b)
        """)
        tree = ast.parse(source)
        transformer = RefactoringTransformer(
            [{"type": "reduce_complexity"}],
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # The match arms' literal patterns MUST still be ``MatchValue``
        # with a ``Constant`` value (not a ``Name``), otherwise the
        # arm has been silently re-interpreted as a capture pattern.
        func_def = new_tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        match_stmt = next(s for s in func_def.body if isinstance(s, ast.Match))
        for case in match_stmt.cases:
            if isinstance(case.pattern, ast.MatchValue):
                assert isinstance(
                    case.pattern.value, ast.Constant
                ), "match value pattern was rewritten — would silently change semantics."

        # Output must compile and dispatch on the integer 42 (not bind).
        compiled = compile(new_tree, "<test-match>", "exec")
        namespace: dict[str, object] = {}
        exec(compiled, namespace)  # noqa: S102
        assert namespace["classify"](42) == ("answer", 84)  # type: ignore[operator]
        assert namespace["classify"](100) == ("century", 84)  # type: ignore[operator]
        assert namespace["classify"](7) == ("other", 84)  # type: ignore[operator]


# ============================================================================
# Learnable3R.fit() tests
# ============================================================================


class TestLearnable3RFit:
    """Tests for Learnable3REngine.fit() with best-epoch checkpointing."""

    @pytest.fixture
    def engine(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from omni_mercury_engine.core.three_r.learnable_fusion import (
            Learnable3RConfig,
            Learnable3REngine,
        )

        # Seed PyTorch in the fixture so weight init / dropout / etc. are
        # deterministic across the training tests.  The NumPy-side
        # ``seed=`` argument passed to ``fit()`` only controls shuffling
        # of the train/val split; PyTorch tensor allocation, parameter
        # initialization, and (when configured) dropout draws all consume
        # the torch RNG.  Without seeding torch here, loss-trajectory
        # assertions in tests below would be flaky across machines and
        # PyTorch versions.
        #
        # NOTE: ``tests/conftest.py::set_random_seed`` is autouse and
        # also seeds torch with ``DEFAULT_TEST_SEED = 42`` before each
        # test runs.  We re-seed inside this fixture defensively so the
        # contract is local to this test class and survives any future
        # change to the global fixture order.
        torch.manual_seed(42)

        config = Learnable3RConfig(hidden_dim=16)
        return Learnable3REngine(config=config, device="cpu")

    def test_fit_returns_history(self, engine: Any) -> None:
        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)

        # ``patience`` is set to ``epochs * 2`` so that early stopping is
        # structurally impossible for this test — the patience counter
        # cannot reach the patience threshold within ``epochs`` epochs.
        # This lets us assert the exact history length deterministically.
        # If a future maintainer lowers ``patience`` below ``epochs``, the
        # universal upper-bound assertion (``<= epochs``) below still
        # holds, and the equality assertion would surface the change as a
        # clear test failure rather than as silent flakiness.
        epochs = 5
        history = engine.fit(X, y, epochs=epochs, batch_size=16, patience=epochs * 2, seed=42)

        assert "train_losses" in history
        assert "val_losses" in history
        assert "best_epoch" in history
        assert "best_val_loss" in history
        assert "stopped_early" in history
        # Universal contract: ``fit()`` never produces more entries than
        # ``epochs``.  This assertion remains correct regardless of
        # patience configuration.
        assert len(history["train_losses"]) <= epochs
        assert len(history["val_losses"]) <= epochs
        # Stronger contract under ``patience >= epochs``: history is
        # exactly ``epochs`` entries long because early stop cannot fire.
        assert len(history["train_losses"]) == epochs
        assert len(history["val_losses"]) == epochs
        assert len(history["train_losses"]) == len(history["val_losses"])
        # And ``stopped_early`` must reflect that.
        assert history["stopped_early"] is False

    def test_fit_loss_decreases(self, engine: Any) -> None:
        # Generate a simple learnable pattern
        rng = np.random.default_rng(123)
        X = rng.standard_normal((100, 4)).astype(np.float32)
        y = (X[:, 0] * 0.5 + X[:, 1] * 0.3).astype(np.float32)

        history = engine.fit(X, y, epochs=30, batch_size=16, patience=20, seed=42)

        # First loss should be higher than last
        assert history["train_losses"][0] > history["train_losses"][-1]

    def test_early_stopping(self, engine: Any) -> None:
        # Random data has no learnable signal, so val_loss plateaus quickly.
        # min_delta=0.01 ensures the patience counter actually advances once
        # improvements become small — at min_delta=1e-10 the test was
        # vulnerable to ever-tinier "improvements" preventing stop.
        rng = np.random.default_rng(seed=42)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        y = rng.standard_normal(50).astype(np.float32)

        history = engine.fit(
            X,
            y,
            epochs=1000,
            batch_size=16,
            patience=3,
            min_delta=0.01,
            seed=42,
        )

        # Should stop well before 1000 epochs and the history length must
        # mirror the early-stop flag.
        assert len(history["train_losses"]) < 1000
        assert history["stopped_early"] is True
        assert len(history["train_losses"]) == len(history["val_losses"])

    def test_best_epoch_checkpoint_restored(self, engine: Any) -> None:
        X = np.random.randn(80, 4).astype(np.float32)
        y = np.random.randn(80).astype(np.float32)

        history = engine.fit(X, y, epochs=20, batch_size=16, patience=5, seed=42)

        # The model should now have the weights from best_epoch, not last epoch
        # We verify by checking the history is well-formed
        best_epoch = history["best_epoch"]
        assert history["val_losses"][best_epoch] == history["best_val_loss"]

    def test_fit_requires_minimum_samples(self, engine: Any) -> None:
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        y = np.array([1.0])

        with pytest.raises(ValueError, match="at least 2 samples"):
            engine.fit(X, y)

    def test_fit_without_pytorch(self) -> None:
        """Test that fit() gracefully handles missing PyTorch."""
        from omni_mercury_engine.core.three_r.learnable_fusion import (
            Learnable3REngine,
        )

        # Create an engine and force model to None.  Learnable3REngine.model
        # is normally ``Learnable3RFusion`` but the engine's runtime guard
        # path (line 592) explicitly tolerates ``None``, so injecting
        # ``None`` here exercises that branch.  The ``type: ignore`` mirrors
        # the matching pragma in the source (``learnable_fusion.py:570``).
        engine = Learnable3REngine.__new__(Learnable3REngine)
        engine.model = None  # type: ignore[assignment]

        history = engine.fit(np.zeros((10, 4)), np.zeros(10))
        assert history["train_losses"] == []
        assert history["stopped_early"] is False


# ============================================================================
# GOSNN AttentionProvider tests
# ============================================================================


class TestAttentionProvider:
    """Tests for the AttentionProvider interface."""

    def test_interface_exists(self) -> None:
        from omni_mercury_engine.core.gosnn_optimizer import AttentionProvider

        assert hasattr(AttentionProvider, "get_attention")

    def test_custom_provider_plugs_in(self) -> None:
        from omni_mercury_engine.core.gosnn_optimizer import (
            AttentionProvider,
            GOSNNOptimizer,
        )

        class MockProvider(AttentionProvider):
            def __init__(self) -> None:
                self.called = False

            def get_attention(self) -> np.ndarray:
                self.called = True
                return np.ones((32, 16, 16))

        provider = MockProvider()
        optimizer = GOSNNOptimizer(attention_provider=provider)
        assert optimizer._attention_provider is provider

    def test_placeholder_warning_when_no_provider(self, caplog: Any) -> None:
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )
        from omni_mercury_engine.core.gosnn_optimizer import GOSNNOptimizer

        reset_global_network()
        gosnn = GlobalOmniScalarNetwork()
        optimizer = GOSNNOptimizer()
        assert optimizer._attention_provider is None

        with caplog.at_level(logging.WARNING, logger="omni_mercury_engine.core.gosnn_optimizer"):
            optimizer.optimize(gosnn)

        assert any(
            "AttentionProvider" in r.getMessage() and r.levelno == logging.WARNING
            for r in caplog.records
        ), "Expected a WARNING mentioning AttentionProvider when none is configured."


# ============================================================================
# BenchmarkDiagnostics logger conversion tests
# ============================================================================


class TestBenchmarkDiagnosticsLogging:
    """Tests that quick_diagnose uses logger, not print."""

    def test_quick_diagnose_uses_logger(self, caplog: Any) -> None:
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

    def test_quick_diagnose_f1_zero_warning(self, caplog: Any) -> None:
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
