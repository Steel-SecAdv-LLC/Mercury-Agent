# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

from omni_mercury_engine.ethical.ethical_alignment_engine import (
    AlignmentArchetype,
    GeometricPatternProcessor,
    IndivisibleEngine,
    PercipienceEngine,
    StrategicEngine,
)


def test_alignment_archetype_is_enum() -> None:
    assert hasattr(AlignmentArchetype, "BALANCE")
    assert hasattr(AlignmentArchetype, "STRATEGY")


def test_indivisible_engine_instantiation() -> None:
    engine = IndivisibleEngine()
    assert engine is not None
    assert callable(getattr(engine, "evaluate_ethical_balance", None))


def test_strategic_engine_instantiation() -> None:
    engine = StrategicEngine()
    assert engine is not None
    assert callable(getattr(engine, "compute_wisdom_quotient", None))


def test_geometric_pattern_processor_instantiation() -> None:
    proc = GeometricPatternProcessor()
    assert proc is not None
    assert callable(getattr(proc, "analyze_geometric_patterns", None))


def test_percipience_engine_instantiation() -> None:
    engine = PercipienceEngine()
    assert engine is not None
    assert callable(getattr(engine, "comprehensive_analysis", None))


def test_importable_from_ethical() -> None:
    from omni_mercury_engine.ethical import (
        AlignmentArchetype as AA,
        IndivisibleEngine as IE,
    )

    assert AA is AlignmentArchetype
    assert IE is IndivisibleEngine
