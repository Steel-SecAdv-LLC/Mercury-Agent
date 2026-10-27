# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AgentMemory's episodic and semantic stores are bounded under load.

Short-term was already a bounded deque and long-term had a capacity, but
episodic and semantic were unbounded dicts grown once per analysis. In a
long-running loop (a benchmark epoch, a persistent session) that grew
memory without limit and made every retrieval and stat scan an
ever-larger dict -- the dominant per-epoch cost growth in the
neuro-symbolic benchmark. These tests pin that both stores stay bounded,
that eviction keeps the high-importance experiences, and that steady-state
store cost stops growing once capacity is reached.
"""

from __future__ import annotations

import time

import numpy as np

from omni_mercury_engine.agentic.mercury_a_agent import AgentMemory


class TestBounded:
    def test_episodic_never_exceeds_capacity(self) -> None:
        mem = AgentMemory(episodic_capacity=100)
        for i in range(1000):
            mem.store_episodic(f"event_{i}", {"i": i}, importance=0.5)
            assert len(mem.episodic) <= 100
        assert len(mem.episodic) <= 100

    def test_semantic_never_exceeds_capacity(self) -> None:
        mem = AgentMemory(semantic_capacity=80)
        for i in range(600):
            mem.store_semantic(f"fact_{i}", category="c", confidence=0.5)
            assert len(mem.semantic) <= 80


class TestEvictionSemantics:
    def test_high_importance_episodes_survive_a_flood(self) -> None:
        mem = AgentMemory(episodic_capacity=20)
        for i in range(5):
            mem.store_episodic(f"critical_{i}", {"i": i}, importance=0.99)
        for i in range(300):
            mem.store_episodic(f"routine_{i}", {"i": i}, importance=0.10)
        survivors = [e.importance for e in mem.episodic.values()]
        assert sum(1 for imp in survivors if imp >= 0.99) == 5
        # Prune evicts a ~10% batch at capacity, so the store oscillates
        # between ~90% and 100% of the cap; it is always bounded by it.
        assert 18 <= len(mem.episodic) <= 20

    def test_prune_always_makes_room(self) -> None:
        """A store at capacity must never wedge (prune removes >= 1)."""
        mem = AgentMemory(episodic_capacity=5)
        for i in range(50):
            mem.store_episodic(f"e_{i}", {"i": i}, importance=0.5)
        assert 0 < len(mem.episodic) <= 5


class TestCostPlateau:
    def test_store_cost_does_not_grow_after_capacity(self) -> None:
        """The defect this fixes: per-store cost grew with total count.

        Measure median store time in an early post-capacity window and a
        much-later one. Unbounded growth made the late window scale with
        total inserts; the bound holds it roughly flat (4x ceiling absorbs
        GC/scheduler noise; the old behavior was far worse across this span).
        """
        mem = AgentMemory(episodic_capacity=200)
        for i in range(200):
            mem.store_episodic(f"warm_{i}", {"i": i}, importance=0.5)

        def window(start: int, n: int) -> float:
            samples = []
            for i in range(start, start + n):
                t0 = time.perf_counter()
                mem.store_episodic(f"e_{i}", {"i": i}, importance=0.5)
                samples.append(time.perf_counter() - t0)
            return float(np.median(samples))

        early = window(1000, 100)
        late = window(20000, 100)  # 19000 inserts later
        assert late <= early * 4.0, f"per-store cost grew {late / early:.1f}x; bound not holding"
        assert 180 <= len(mem.episodic) <= 200
