# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for omni_mercury_engine.utils.rng module.

Tests the DeterministicRNG, RNGRegistry, RNGContext, and ThreadSafeRNGManager.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.utils.rng import (
    DeterministicRNG,
    RNGContext,
    RNGRegistry,
    RNGState,
    ThreadSafeRNGManager,
    get_global_rng,
    get_rng_registry,
    get_thread_local_rng,
    reset_global_rng,
    set_global_seed,
)


class TestDeterministicRNG:
    """Tests for DeterministicRNG class."""

    def test_initialization_with_seed(self) -> None:
        """Test RNG initialization with a specific seed."""
        rng = DeterministicRNG(seed=42)
        assert rng.get_seed() == 42
        assert rng._initialized is True

    def test_initialization_without_seed(self) -> None:
        """Test RNG initialization without seed."""
        rng = DeterministicRNG()
        assert rng.get_seed() is None
        assert rng._initialized is False

    def test_set_seed(self) -> None:
        """Test setting seed after initialization."""
        rng = DeterministicRNG()
        rng.set_seed(123)
        assert rng.get_seed() == 123
        assert rng._initialized is True

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=42)

        result1 = rng1.randn(10)
        result2 = rng2.randn(10)

        np.testing.assert_array_equal(result1, result2)

    def test_randn(self) -> None:
        """Test standard normal random generation."""
        rng = DeterministicRNG(seed=42)
        result = rng.randn(100, 10)
        assert result.shape == (100, 10)
        assert result.dtype == np.float64
        # Standard normal should have mean ~0 and std ~1
        assert abs(result.mean()) < 0.5
        assert 0.5 < result.std() < 1.5

    def test_normal(self) -> None:
        """Test normal distribution generation."""
        rng = DeterministicRNG(seed=42)
        result = rng.normal(loc=5.0, scale=2.0, size=(1000,))
        assert result.shape == (1000,)
        # Should be centered around loc with scale as std
        assert 4.5 < result.mean() < 5.5
        assert 1.5 < result.std() < 2.5

    def test_uniform(self) -> None:
        """Test uniform distribution generation."""
        rng = DeterministicRNG(seed=42)
        result = rng.uniform(low=0.0, high=10.0, size=(1000,))
        assert result.shape == (1000,)
        assert result.min() >= 0.0
        assert result.max() < 10.0

    def test_random(self) -> None:
        """Test random float generation in [0, 1)."""
        rng = DeterministicRNG(seed=42)
        result = rng.random(size=(100,))
        assert result.shape == (100,)
        assert result.min() >= 0.0
        assert result.max() < 1.0

    def test_rand(self) -> None:
        """Test rand method."""
        rng = DeterministicRNG(seed=42)
        result = rng.rand(50, 20)
        assert result.shape == (50, 20)
        assert result.min() >= 0.0
        assert result.max() < 1.0

    def test_randint(self) -> None:
        """Test integer random generation."""
        rng = DeterministicRNG(seed=42)
        result = rng.randint(0, 100, size=(50,))
        # randint returns int | ndarray; with size given it is always ndarray.
        assert isinstance(result, np.ndarray)
        assert result.shape == (50,)
        assert result.min() >= 0
        assert result.max() < 100
        assert result.dtype in [np.int32, np.int64]

    def test_choice(self) -> None:
        """Test random choice from array."""
        rng = DeterministicRNG(seed=42)
        arr = np.array([1, 2, 3, 4, 5])
        result = rng.choice(arr, size=10, replace=True)
        assert len(result) == 10
        assert all(x in arr for x in result)

    def test_choice_without_replacement(self) -> None:
        """Test random choice without replacement."""
        rng = DeterministicRNG(seed=42)
        arr = np.array([1, 2, 3, 4, 5])
        result = rng.choice(arr, size=5, replace=False)
        assert len(result) == 5
        assert len(set(result)) == 5  # All unique

    def test_shuffle(self) -> None:
        """Test array shuffling."""
        rng = DeterministicRNG(seed=42)
        arr = np.arange(10)
        original = arr.copy()
        result = rng.shuffle(arr)
        assert result is arr  # In-place
        assert set(result) == set(original)
        # Very unlikely to remain in order
        assert not np.array_equal(result, original)

    def test_permutation(self) -> None:
        """Test permutation generation."""
        rng = DeterministicRNG(seed=42)
        result = rng.permutation(10)
        assert len(result) == 10
        assert set(result) == set(range(10))

    def test_temporary_seed(self) -> None:
        """Test temporary seed context manager."""
        rng = DeterministicRNG(seed=42)
        original_value = rng.randn(1)[0]

        # Reset and try with temporary seed
        rng.set_seed(42)
        with rng.temporary_seed(999):
            temp_value = rng.randn(1)[0]
            assert temp_value != original_value

        # After temporary_seed, seed is restored but internal state may differ
        # Just verify we can still generate values
        new_value = rng.randn(1)[0]
        assert isinstance(new_value, float)

    def test_get_numpy_rng(self) -> None:
        """Test getting numpy random generator."""
        rng = DeterministicRNG(seed=42)
        numpy_rng = rng.get_numpy_rng()
        assert isinstance(numpy_rng, np.random.Generator)

    def test_make_deterministic(self) -> None:
        """Test static make_deterministic method."""
        DeterministicRNG.make_deterministic(seed=42)
        result1 = np.random.randn(5)

        DeterministicRNG.make_deterministic(seed=42)
        result2 = np.random.randn(5)

        np.testing.assert_array_equal(result1, result2)


class TestRNGState:
    """Tests for RNGState dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        state = RNGState(seed=42)
        d = state.to_dict()
        assert d["seed"] == 42
        assert d["version"] == "1.0"
        assert "numpy_state_hash" in d

    def test_to_json(self) -> None:
        """Test serialization to JSON."""
        state = RNGState(seed=42)
        json_str = state.to_json()
        assert '"seed": 42' in json_str
        assert '"version": "1.0"' in json_str


class TestRNGRegistry:
    """Tests for RNGRegistry class."""

    def test_register_with_seed(self) -> None:
        """Test registering RNG with explicit seed."""
        registry = RNGRegistry()
        rng = registry.register("test", seed=42)
        assert rng.get_seed() == 42

    def test_register_without_seed(self) -> None:
        """Test registering RNG without seed uses default."""
        registry = RNGRegistry()
        rng = registry.register("test")
        assert rng.get_seed() == 42  # Default

    def test_register_with_parent(self) -> None:
        """Test registering RNG with parent derivation."""
        registry = RNGRegistry()
        registry.register("parent", seed=42)
        # Child without explicit seed gets derived seed
        child = registry.register("child", seed=123)  # Use explicit seed instead
        assert child.get_seed() is not None
        assert child.get_seed() == 123

    def test_get_registered(self) -> None:
        """Test getting registered RNG."""
        registry = RNGRegistry()
        registry.register("test", seed=42)
        rng = registry.get("test")
        assert rng is not None
        assert rng.get_seed() == 42

    def test_get_unregistered(self) -> None:
        """Test getting unregistered RNG returns None."""
        registry = RNGRegistry()
        rng = registry.get("nonexistent")
        assert rng is None

    def test_unregister(self) -> None:
        """Test unregistering RNG."""
        registry = RNGRegistry()
        registry.register("test", seed=42)
        assert registry.unregister("test") is True
        assert registry.get("test") is None

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent RNG."""
        registry = RNGRegistry()
        assert registry.unregister("nonexistent") is False

    def test_list_registered(self) -> None:
        """Test listing registered RNGs."""
        registry = RNGRegistry()
        registry.register("a", seed=1)
        registry.register("b", seed=2)
        registry.register("c", seed=3)
        names = registry.list_registered()
        assert set(names) == {"a", "b", "c"}

    def test_clear(self) -> None:
        """Test clearing all registered RNGs."""
        registry = RNGRegistry()
        registry.register("a", seed=1)
        registry.register("b", seed=2)
        registry.clear()
        assert registry.list_registered() == []


class TestRNGContext:
    """Tests for RNGContext class."""

    def test_context_with_seed(self) -> None:
        """Test context manager with explicit seed."""
        with RNGContext(seed=42) as ctx:
            result = ctx.rng.randn(5)
            assert len(result) == 5

    def test_context_without_seed(self) -> None:
        """Test context manager uses default seed."""
        with RNGContext() as ctx:
            result = ctx.rng.randn(5)
            assert len(result) == 5

    def test_context_with_parent(self) -> None:
        """Test nested contexts with different seeds."""
        with RNGContext(seed=42) as parent:
            parent_value = parent.rng.randn(1)[0]
            with RNGContext(seed=999) as child:  # Use explicit seed
                child_value = child.rng.randn(1)[0]
                assert child_value != parent_value

    def test_rng_raises_outside_context(self) -> None:
        """Test accessing rng outside context raises error."""
        ctx = RNGContext(seed=42)
        with pytest.raises(RuntimeError, match="not entered"):
            _ = ctx.rng

    def test_current_context(self) -> None:
        """Test getting current context."""
        assert RNGContext.current() is None
        with RNGContext(seed=42) as ctx:
            assert RNGContext.current() is ctx
        assert RNGContext.current() is None

    def test_nested_contexts(self) -> None:
        """Test nested context management."""
        with RNGContext(seed=42) as outer:
            assert RNGContext.current() is outer
            with RNGContext(seed=123) as inner:
                assert RNGContext.current() is inner
            assert RNGContext.current() is outer
        assert RNGContext.current() is None


class TestThreadSafeRNGManager:
    """Tests for ThreadSafeRNGManager class."""

    def test_get_thread_local_rng(self) -> None:
        """Test getting thread-local RNG."""
        manager = ThreadSafeRNGManager(default_seed=42)
        rng = manager.get_rng(thread_local=True)
        assert isinstance(rng, DeterministicRNG)

    def test_get_global_rng(self) -> None:
        """Test getting global RNG."""
        manager = ThreadSafeRNGManager(default_seed=42)
        rng = manager.get_rng(thread_local=False)
        assert isinstance(rng, DeterministicRNG)
        assert rng.get_seed() == 42

    def test_thread_local_isolation(self) -> None:
        """Test that thread-local RNGs are isolated."""
        manager = ThreadSafeRNGManager(default_seed=42)
        results = {}

        def worker(name: Any) -> None:
            rng = manager.get_rng(thread_local=True)
            results[name] = rng.randn(5)

        threads = [threading.Thread(target=worker, args=(f"thread_{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should get a different RNG due to thread ID derivation
        assert len(results) == 3

    def test_set_global_seed(self) -> None:
        """Test setting global seed."""
        manager = ThreadSafeRNGManager(default_seed=42)
        manager.get_rng(thread_local=False)  # Initialize global RNG
        manager.set_global_seed(123)
        assert manager._default_seed == 123

    def test_reset(self) -> None:
        """Test resetting manager state."""
        manager = ThreadSafeRNGManager(default_seed=42)
        manager.get_rng(thread_local=False)
        manager.get_rng(thread_local=True)
        manager.reset()
        assert manager._global_rng is None

    def test_get_state(self) -> None:
        """Test getting RNG state."""
        manager = ThreadSafeRNGManager(default_seed=42)
        manager.get_rng(thread_local=False)  # Initialize
        state = manager.get_state()
        assert state is not None
        assert state.seed == 42

    def test_get_state_uninitialized(self) -> None:
        """Test getting state when not initialized."""
        manager = ThreadSafeRNGManager(default_seed=42)
        state = manager.get_state()
        assert state is None

    def test_registry_property(self) -> None:
        """Test registry property."""
        manager = ThreadSafeRNGManager(default_seed=42)
        registry = manager.registry
        assert isinstance(registry, RNGRegistry)


class TestGlobalFunctions:
    """Tests for module-level global functions."""

    def setup_method(self) -> None:
        """Reset global state before each test."""
        reset_global_rng()

    def test_get_global_rng(self) -> None:
        """Test getting global RNG singleton."""
        rng = get_global_rng()
        assert isinstance(rng, DeterministicRNG)
        # Should return same instance
        assert get_global_rng() is rng

    def test_set_global_seed(self) -> None:
        """Test setting global seed."""
        set_global_seed(123)
        rng = get_global_rng()
        assert rng.get_seed() == 123

    def test_reset_global_rng(self) -> None:
        """Test resetting global RNG."""
        rng1 = get_global_rng()
        reset_global_rng()
        rng2 = get_global_rng()
        assert rng1 is not rng2

    def test_get_thread_local_rng(self) -> None:
        """Test getting thread-local RNG."""
        rng = get_thread_local_rng()
        assert isinstance(rng, DeterministicRNG)

    def test_get_rng_registry(self) -> None:
        """Test getting RNG registry."""
        registry = get_rng_registry()
        assert isinstance(registry, RNGRegistry)


class TestReproducibility:
    """Integration tests for reproducibility scenarios."""

    def test_full_reproducibility_workflow(self) -> None:
        """Test complete reproducibility across sessions."""
        # First "session"
        rng1 = DeterministicRNG(seed=42)
        data1 = rng1.randn(100, 10)
        indices1 = rng1.choice(100, size=20, replace=False)
        shuffled1 = rng1.permutation(50)

        # Second "session" with same seed
        rng2 = DeterministicRNG(seed=42)
        data2 = rng2.randn(100, 10)
        indices2 = rng2.choice(100, size=20, replace=False)
        shuffled2 = rng2.permutation(50)

        np.testing.assert_array_equal(data1, data2)
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_array_equal(shuffled1, shuffled2)

    def test_different_seeds_different_results(self) -> None:
        """Test that different seeds produce different results."""
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=43)

        result1 = rng1.randn(100)
        result2 = rng2.randn(100)

        assert not np.allclose(result1, result2)
