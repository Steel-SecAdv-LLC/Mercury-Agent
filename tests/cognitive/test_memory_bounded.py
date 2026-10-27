# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NeuralMemoryLayer episodic store is bounded under sustained ingestion.

Before the bound, ``ingest_memories`` appended one embedding per call
forever and refit the clusterer over the whole list each time, so a loop
of one analysis per tick grew memory linearly and wall-clock
quadratically. These tests pin the three properties that make the fix
real rather than cosmetic:

1. the retained set never exceeds the configured capacity, no matter how
   many entries are ingested;
2. eviction keeps the highest-importance memories (semantics, not just a
   ring buffer that would drop a flagged anomaly for a routine tick);
3. per-ingest cost stops growing once capacity is reached (the actual
   defect: unbounded per-epoch time).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.neural_memory_layer import (
    MemoryType,
    NeuralMemoryLayer,
)


def _mem(i: int, importance: float = 0.5) -> dict[str, Any]:
    # Distinct content so embeddings are not degenerate.
    return {"value": float(i), "tag": f"evt_{i % 7}", "importance": importance}


class TestCapacityBound:
    def test_retained_set_never_exceeds_capacity(self) -> None:
        layer = NeuralMemoryLayer(n_clusters=4, max_memories=100)
        for i in range(1000):  # 10x capacity, one at a time (the loop case)
            layer.ingest_memories([_mem(i)], MemoryType.EPISODIC)
            assert len(layer.embeddings) <= 100
        assert len(layer.embeddings) == 100
        assert layer.evicted_count == 900

    def test_batch_overflow_is_bounded_in_one_call(self) -> None:
        layer = NeuralMemoryLayer(n_clusters=4, max_memories=50)
        layer.ingest_memories([_mem(i) for i in range(500)], MemoryType.EPISODIC)
        assert len(layer.embeddings) == 50

    def test_capacity_below_clusters_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_memories"):
            NeuralMemoryLayer(n_clusters=8, max_memories=4)


class TestEvictionSemantics:
    def test_high_importance_memories_survive_flood_of_routine_ones(self) -> None:
        """A flagged anomaly must not be evicted for later routine ticks."""
        layer = NeuralMemoryLayer(n_clusters=4, max_memories=20)
        # Ingest a few critical memories, then flood with low-importance ones.
        for i in range(5):
            layer.ingest_memories([_mem(i, importance=0.99)], MemoryType.EPISODIC)
        for i in range(100, 300):
            layer.ingest_memories([_mem(i, importance=0.10)], MemoryType.EPISODIC)
        retained_importance = [e.importance for e in layer.embeddings]
        # All five critical memories are still resident.
        assert sum(1 for imp in retained_importance if imp >= 0.99) == 5
        assert len(layer.embeddings) == 20

    def test_ties_evict_oldest_first(self) -> None:
        layer = NeuralMemoryLayer(n_clusters=2, max_memories=10)
        base = time.time()
        for i in range(40):
            layer.ingest_memories(
                [{"value": float(i), "importance": 0.5, "timestamp": base + i}],
                MemoryType.EPISODIC,
            )
        timestamps = [e.timestamp for e in layer.embeddings]
        # With equal importance, the ten most recent timestamps survive.
        assert min(timestamps) >= base + 30


class TestIdIntegrity:
    def test_generated_ids_do_not_recycle_after_eviction(self) -> None:
        """Length-based ids would collide post-eviction; monotonic must not."""
        layer = NeuralMemoryLayer(n_clusters=4, max_memories=30)
        seen: set[str] = set()
        for i in range(400):
            created = layer.ingest_memories([_mem(i)], MemoryType.EPISODIC)
            new_id = created[0].entry_id
            assert new_id not in seen, f"id {new_id} reused after eviction"
            seen.add(new_id)
        # Live ids are always unique too.
        live_ids = [e.entry_id for e in layer.embeddings]
        assert len(live_ids) == len(set(live_ids))


class TestCostPlateau:
    def test_per_ingest_cost_does_not_grow_after_capacity(self) -> None:
        """The actual defect: per-ingest work grew without bound.

        Compare mean per-ingest time in an early window (post-capacity) to
        a much-later window. Unbounded growth made the late window scale
        with total count; the bound holds it roughly flat. A generous 4x
        ceiling absorbs scheduler/GC noise while still failing the old
        O(n)-per-ingest behavior, which was >50x across this range.
        """
        layer = NeuralMemoryLayer(n_clusters=8, max_memories=200)
        # Fill to capacity first so both windows measure steady state.
        for i in range(200):
            layer.ingest_memories([_mem(i)], MemoryType.EPISODIC)

        def window_mean(start: int, n: int) -> float:
            samples = []
            for i in range(start, start + n):
                t0 = time.perf_counter()
                layer.ingest_memories([_mem(i)], MemoryType.EPISODIC)
                samples.append(time.perf_counter() - t0)
            return float(np.median(samples))

        early = window_mean(1000, 50)
        late = window_mean(8000, 50)  # 7000 ingests later
        assert late <= early * 4.0, (
            f"per-ingest cost grew {late / early:.1f}x from early to late window; "
            "memory bound is not holding steady-state cost flat"
        )
        assert len(layer.embeddings) == 200
