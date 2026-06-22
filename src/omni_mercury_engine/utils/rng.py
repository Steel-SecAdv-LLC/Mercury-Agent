# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Internal Random Number Generator Utility for Test Determinism.

Provides centralized RNG management with seed control for reproducible testing.

Thread-Safety Features:
- Thread-local storage for per-thread RNG instances
- Lock-protected global operations
- RNG registry pattern for named generators
- Hierarchical state management with RNGContext
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

# Make torch optional to support environments without ML dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment,unused-ignore]
    TORCH_AVAILABLE = False


class DeterministicRNG:
    """Centralized random number generator for test determinism.

    Features:
    - Single source of truth for random state
    - Easy seed control for reproducibility
    - Context manager for temporary seeding
    - Support for NumPy, Python random, and PyTorch
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the RNG with optional seed.

        Args:
            seed: Random seed for reproducibility. If None, uses random initialization.
        """
        self._seed = seed
        self._numpy_rng: np.random.Generator | None = None
        self._initialized = False

        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Set the random seed for all RNG backends.

        Args:
            seed: Random seed value
        """
        self._seed = seed

        np.random.seed(seed)

        random.seed(seed)

        if TORCH_AVAILABLE and torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        self._numpy_rng = np.random.default_rng(seed)
        self._initialized = True

    def get_numpy_rng(self) -> np.random.Generator:
        """Get the NumPy random generator.

        Returns:
            NumPy Generator instance

        Raises:
            RuntimeError: If RNG is not initialized after set_seed call
        """
        if not self._initialized:
            self.set_seed(self._seed or 42)
        if self._numpy_rng is None:
            raise RuntimeError("NumPy RNG not initialized after set_seed call")
        return self._numpy_rng

    def randn(self, *shape: int, dtype: type = np.float64) -> np.ndarray[Any, Any]:
        """Generate standard normal random numbers.

        Args:
            *shape: Shape of output array
            dtype: Data type for output

        Returns:
            Random array from standard normal distribution
        """
        rng = self.get_numpy_rng()
        return rng.standard_normal(size=shape).astype(dtype)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: int | tuple[int, ...] | None = None,
        dtype: type = np.float64,
    ) -> np.ndarray[Any, Any]:
        """Draw random numbers from the Gaussian (normal) distribution.

        Args:
            loc: Mean of the distribution
            scale: Standard deviation of the distribution
            size: Output shape
            dtype: Data type for output

        Returns:
            Random array from normal distribution
        """
        rng = self.get_numpy_rng()
        return np.asarray(rng.normal(loc=loc, scale=scale, size=size)).astype(dtype)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
        dtype: type = np.float64,
    ) -> np.ndarray[Any, Any]:
        """Generate random numbers from uniform distribution.

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)
            size: Output shape
            dtype: Data type for output

        Returns:
            Random array from uniform distribution
        """
        rng = self.get_numpy_rng()
        return np.asarray(rng.uniform(low=low, high=high, size=size)).astype(dtype)

    def random(self, size: int | tuple[int, ...] | None = None) -> np.ndarray[Any, Any]:
        """Generate random floats in the half-open interval [0.0, 1.0).

        Args:
            size: Output shape

        Returns:
            Random array from [0.0, 1.0)
        """
        rng = self.get_numpy_rng()
        result = rng.random(size=size)
        return np.asarray(result)

    def rand(self, *shape: int, dtype: type = np.float64) -> np.ndarray[Any, Any]:
        """Generate uniform random numbers in [0, 1).

        Args:
            *shape: Shape of output array
            dtype: Data type for output

        Returns:
            Random array from uniform distribution
        """
        rng = self.get_numpy_rng()
        return rng.random(size=shape).astype(dtype)

    def randint(
        self, low: int, high: int | None = None, size: int | tuple[int, ...] | None = None
    ) -> int | np.ndarray[Any, Any]:
        """Generate random integers.

        Args:
            low: Lower bound (inclusive) or if high is None, upper bound (exclusive)
            high: Upper bound (exclusive)
            size: Output shape

        Returns:
            Random integer(s)
        """
        rng = self.get_numpy_rng()
        return rng.integers(low=low, high=high, size=size)

    def choice(
        self,
        a: int | np.ndarray[Any, Any],
        size: int | tuple[int, ...] | None = None,
        replace: bool = True,
        p: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Generate random samples from array.

        Args:
            a: Array to sample from, or int for range(a)
            size: Output shape
            replace: Whether to sample with replacement
            p: Probabilities for each element

        Returns:
            Random sample(s)
        """
        rng = self.get_numpy_rng()
        return np.asarray(rng.choice(a, size=size, replace=replace, p=p))

    def shuffle(self, array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Shuffle array in-place.

        Args:
            array: Array to shuffle

        Returns:
            Shuffled array (same object as input)
        """
        rng = self.get_numpy_rng()
        rng.shuffle(array)
        return array

    def permutation(self, x: int | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Generate random permutation.

        Args:
            x: Array to permute, or int for range(x)

        Returns:
            Permuted array
        """
        rng = self.get_numpy_rng()
        return rng.permutation(x)

    @contextmanager
    def temporary_seed(self, seed: int) -> Generator[Any, None, None]:
        """Context manager for temporary seed override.

        Example:
            with rng.temporary_seed(123):
                data = rng.randn(100, 10)

        Args:
            seed: Temporary seed value
        """
        old_seed = self._seed
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()
        old_torch_state = None
        if TORCH_AVAILABLE and torch is not None:
            old_torch_state = torch.get_rng_state()

        self.set_seed(seed)

        try:
            yield self
        finally:
            self._seed = old_seed
            np.random.set_state(old_numpy_state)
            random.setstate(old_random_state)
            if TORCH_AVAILABLE and torch is not None and old_torch_state is not None:
                torch.set_rng_state(old_torch_state)
            if self._seed is not None:
                self._numpy_rng = np.random.default_rng(self._seed)

    def get_seed(self) -> int | None:
        """Get the current seed value.

        Returns:
            Current seed or None if not set
        """
        return self._seed

    @staticmethod
    def make_deterministic(seed: int = 42) -> None:
        """Make all random operations deterministic across the entire environment.

        Args:
            seed: Seed value for determinism
        """
        np.random.seed(seed)
        random.seed(seed)

        if TORCH_AVAILABLE and torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


@dataclass
class RNGState:
    """Serializable RNG state for reproducibility across processes."""

    seed: int
    numpy_state: dict[str, Any] | None = None
    python_state: tuple[Any, ...] | None = None
    torch_state: bytes | None = None
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "seed": self.seed,
            "version": self.version,
            "numpy_state_hash": (
                # Using SHA3-256 for AMA Cryptography alignment
                # Note: This is non-cryptographic use for state hashing
                hashlib.sha3_256(str(self.numpy_state).encode()).hexdigest()
                if self.numpy_state
                else None
            ),
        }

    def to_json(self) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.to_dict())


class RNGRegistry:
    """Registry pattern for named RNG generators.

    Provides centralized management of multiple RNG instances with
    thread-safe access and hierarchical seed derivation.

    Example:
        registry = RNGRegistry()
        registry.register("training", seed=42)
        registry.register("validation", seed=123)

        train_rng = registry.get("training")
        val_rng = registry.get("validation")
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self._registry: dict[str, DeterministicRNG] = {}
        self._lock = threading.RLock()

    def register(
        self, name: str, seed: int | None = None, parent: str | None = None
    ) -> DeterministicRNG:
        """Register a new named RNG.

        Args:
            name: Unique name for the RNG
            seed: Seed value (derived from parent if not provided)
            parent: Optional parent RNG name for hierarchical seeding

        Returns:
            The registered DeterministicRNG instance
        """
        with self._lock:
            if seed is None and parent and parent in self._registry:
                # Derive seed from parent
                parent_rng = self._registry[parent]
                seed = int(parent_rng.randint(0, 2**31 - 1))
            elif seed is None:
                seed = 42

            rng = DeterministicRNG(seed=seed)
            self._registry[name] = rng
            return rng

    def get(self, name: str) -> DeterministicRNG | None:
        """Get a registered RNG by name."""
        with self._lock:
            return self._registry.get(name)

    def unregister(self, name: str) -> bool:
        """Unregister and remove an RNG."""
        with self._lock:
            if name in self._registry:
                del self._registry[name]
                return True
            return False

    def list_registered(self) -> list[Any]:
        """List all registered RNG names."""
        with self._lock:
            return list(self._registry.keys())

    def clear(self) -> None:
        """Clear all registered RNGs."""
        with self._lock:
            self._registry.clear()


class RNGContext:
    """Hierarchical RNG context manager for scoped state isolation.

    Provides nested RNG scopes where child contexts derive seeds from
    parents, ensuring reproducibility while maintaining isolation.

    Note: Each context creates its own DeterministicRNG instance with a
    derived seed. This provides isolation without modifying global state.

    Example:
        with RNGContext(seed=42) as ctx:
            data1 = ctx.rng.randn(100)
            with RNGContext(parent=ctx) as child_ctx:
                data2 = child_ctx.rng.randn(50)
            # Parent context state preserved
    """

    _context_stack = threading.local()

    def __init__(self, seed: int | None = None, parent: RNGContext | None = None) -> None:
        """Initialize the instance."""
        self._seed = seed
        self._parent = parent
        self._rng: DeterministicRNG | None = None

    @property
    def rng(self) -> DeterministicRNG:
        """Get the RNG for this context."""
        if self._rng is None:
            raise RuntimeError("RNGContext not entered. Use 'with' statement.")
        return self._rng

    def __enter__(self) -> RNGContext:
        # Determine seed
        """Enter the context manager."""
        if self._seed is not None:
            seed = self._seed
        elif self._parent is not None:
            seed = int(self._parent.rng.randint(0, 2**31 - 1))
        else:
            seed = 42

        # Create RNG for this context
        self._rng = DeterministicRNG(seed=seed)

        # Push to context stack
        if not hasattr(self._context_stack, "stack"):
            self._context_stack.stack = []
        self._context_stack.stack.append(self)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        # Pop from context stack
        """Exit the context manager."""
        if hasattr(self._context_stack, "stack") and self._context_stack.stack:
            self._context_stack.stack.pop()
        self._rng = None

    @classmethod
    def current(cls) -> RNGContext | None:
        """Get the current active RNG context."""
        if hasattr(cls._context_stack, "stack") and cls._context_stack.stack:
            result: RNGContext = cls._context_stack.stack[-1]
            return result
        return None


class ThreadSafeRNGManager:
    """Thread-safe global RNG manager using thread-local storage.

    Replaces the global mutable state pattern with dependency injection
    and thread-local instances for safe concurrent access.

    Features:
    - Per-thread RNG instances via thread-local storage
    - Lock-protected global operations
    - RNG pool for lock-free concurrent access
    - State serialization for reproducibility across processes
    """

    def __init__(self, default_seed: int = 42) -> None:
        """Initialize the instance."""
        self._default_seed = default_seed
        self._lock = threading.RLock()
        self._thread_local = threading.local()
        self._registry = RNGRegistry()
        self._global_rng: DeterministicRNG | None = None

    def get_rng(self, thread_local: bool = True) -> DeterministicRNG:
        """Get an RNG instance.

        Args:
            thread_local: If True, returns a thread-local instance.
                         If False, returns the shared global instance.

        Returns:
            DeterministicRNG instance
        """
        if thread_local:
            if not hasattr(self._thread_local, "rng"):
                # Create thread-local RNG with derived seed
                with self._lock:
                    # Derive seed from thread ID for uniqueness
                    thread_seed = (self._default_seed + hash(threading.current_thread().ident)) % (
                        2**31
                    )
                self._thread_local.rng = DeterministicRNG(seed=thread_seed)
            thread_rng: DeterministicRNG = self._thread_local.rng
            return thread_rng
        else:
            with self._lock:
                if self._global_rng is None:
                    self._global_rng = DeterministicRNG(seed=self._default_seed)
                return self._global_rng

    def set_global_seed(self, seed: int) -> None:
        """Set the seed for the global RNG (thread-safe)."""
        with self._lock:
            self._default_seed = seed
            if self._global_rng is not None:
                self._global_rng.set_seed(seed)

    def reset(self) -> None:
        """Reset all RNG state (thread-safe)."""
        with self._lock:
            self._global_rng = None
            self._registry.clear()
        # Clear thread-local storage
        if hasattr(self._thread_local, "rng"):
            del self._thread_local.rng

    def get_state(self) -> RNGState | None:
        """Get serializable state of the global RNG."""
        with self._lock:
            if self._global_rng is None:
                return None

            rng_state = np.random.get_state(legacy=False)
            numpy_state = {
                "bit_generator": rng_state.get("bit_generator", ""),
                "state": rng_state.get("state", {}),
            }

            return RNGState(
                seed=self._global_rng.get_seed() or self._default_seed,
                numpy_state=numpy_state,
                python_state=random.getstate(),
            )

    @property
    def registry(self) -> RNGRegistry:
        """Get the RNG registry for named generators."""
        return self._registry


# Thread-safe singleton manager
_rng_manager: ThreadSafeRNGManager | None = None
_manager_lock = threading.Lock()


def _get_manager() -> ThreadSafeRNGManager:
    """Get or create the singleton RNG manager."""
    global _rng_manager
    if _rng_manager is None:
        with _manager_lock:
            if _rng_manager is None:
                _rng_manager = ThreadSafeRNGManager()
    return _rng_manager


# Legacy API compatibility - now thread-safe
_global_rng: DeterministicRNG | None = None
_global_lock = threading.RLock()


def get_global_rng() -> DeterministicRNG:
    """Get the global RNG instance (thread-safe).

    For new code, prefer using ThreadSafeRNGManager or RNGContext
    for better thread isolation.

    Returns:
        Global DeterministicRNG instance
    """
    global _global_rng
    with _global_lock:
        if _global_rng is None:
            _global_rng = DeterministicRNG(seed=42)
        return _global_rng


def set_global_seed(seed: int) -> None:
    """Set the global random seed (thread-safe).

    Args:
        seed: Random seed value
    """
    global _global_rng
    with _global_lock:
        if _global_rng is None:
            _global_rng = DeterministicRNG(seed=seed)
        else:
            _global_rng.set_seed(seed)


def reset_global_rng() -> None:
    """Reset the global RNG to uninitialized state (thread-safe)."""
    global _global_rng
    with _global_lock:
        _global_rng = None


def get_thread_local_rng() -> DeterministicRNG:
    """Get a thread-local RNG instance.

    This is the preferred method for multi-threaded applications.

    Returns:
        Thread-local DeterministicRNG instance
    """
    return _get_manager().get_rng(thread_local=True)


def get_rng_registry() -> RNGRegistry:
    """Get the RNG registry for named generators.

    Returns:
        Global RNGRegistry instance
    """
    return _get_manager().registry
