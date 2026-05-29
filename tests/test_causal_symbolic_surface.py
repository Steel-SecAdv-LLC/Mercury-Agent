"""
Tests for surfacing the symbolic stack (Issue #4): causal discovery
(PC + Fisher-Z + Granger) and the symbolic rule graph, through the engine
and CLI.

Done-criterion focus: discovery runs deterministically on a fixed seed and
produces a stable graph on a known fixture.

Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")


def _linear_sem(seed: int = 0, n: int = 800) -> np.ndarray:
    """Known structure: A -> B -> C (chain); D independent."""
    rng = np.random.RandomState(seed)
    a = rng.normal(0, 1, n)
    b = 1.5 * a + rng.normal(0, 0.5, n)
    c = -1.2 * b + rng.normal(0, 0.5, n)
    d = rng.normal(0, 1, n)
    return np.column_stack([a, b, c, d])


@pytest.fixture
def engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(mode="fusion", device="cpu")


class TestCausalDiscoverySurface:
    def test_deterministic_on_fixed_seed(self, engine: Any) -> None:
        X = _linear_sem()
        g1 = engine.discover_causal_structure(X, ["A", "B", "C", "D"], seed=42)
        g2 = engine.discover_causal_structure(X, ["A", "B", "C", "D"], seed=42)

        def edges(g: dict) -> list:
            return sorted(
                (e["source"], e["target"], e["type"], round(e["strength"], 6)) for e in g["edges"]
            )

        assert edges(g1) == edges(g2), "causal discovery must be deterministic for a fixed seed"

    def test_recovers_known_skeleton(self, engine: Any) -> None:
        X = _linear_sem()
        g = engine.discover_causal_structure(X, ["A", "B", "C", "D"], seed=0)
        skeleton = {frozenset((e["source"], e["target"])) for e in g["edges"]}

        # The chain edges must be recovered.
        assert frozenset(("A", "B")) in skeleton
        assert frozenset(("B", "C")) in skeleton
        # The independent variable D must not be adjacent to anything.
        for node in ("A", "B", "C"):
            assert frozenset((node, "D")) not in skeleton

    def test_rejects_non_2d(self, engine: Any) -> None:
        with pytest.raises(ValueError, match="2-D"):
            engine.discover_causal_structure(np.zeros((3, 3, 3)))

    def test_temporal_causation_runs(self, engine: Any) -> None:
        # A leads B by one step; deterministic dict output.
        rng = np.random.RandomState(0)
        a = rng.normal(0, 1, 400)
        b = np.concatenate([[0.0], 0.8 * a[:-1]]) + rng.normal(0, 0.3, 400)
        X = np.column_stack([a, b])
        g1 = engine.discover_temporal_causation(X, ["A", "B"], max_lag=3, seed=1)
        g2 = engine.discover_temporal_causation(X, ["A", "B"], max_lag=3, seed=1)
        assert g1["edges"] == g2["edges"]
        assert set(g1["nodes"]) == {"A", "B"}


class TestSymbolicRuleGraphSurface:
    def test_rule_graph_export(self, engine: Any) -> None:
        graph = engine.symbolic_rule_graph()
        assert "statistics" in graph and "rules" in graph
        assert graph["statistics"]["num_rules"] > 0
        assert len(graph["rules"]) == graph["statistics"]["num_rules"]
        sample = graph["rules"][0]
        for key in ("rule_id", "type", "premise", "conclusion", "confidence"):
            assert key in sample

    def test_rule_graph_deterministic(self, engine: Any) -> None:
        assert engine.symbolic_rule_graph() == engine.symbolic_rule_graph()


class TestCausalCLI:
    def test_cli_causal_command(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        X = _linear_sem()
        csv = tmp_path / "sem.csv"
        np.savetxt(csv, X, delimiter=",")

        runner = CliRunner()
        result = runner.invoke(
            main, ["causal", "-i", str(csv), "--names", "A,B,C,D", "--seed", "0"]
        )
        assert result.exit_code == 0, result.output
        assert '"edges"' in result.output

    def test_cli_symbolic_rules_command(self) -> None:
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["symbolic-rules"])
        assert result.exit_code == 0, result.output
        assert '"rules"' in result.output
