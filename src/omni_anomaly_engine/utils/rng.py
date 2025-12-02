"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Internal Random Number Generator Utility for Test Determinism

Provides centralized RNG management with seed control for reproducible testing.
"""

import numpy as np
import random
import torch
from typing import Optional, Union
from contextlib import contextmanager


class DeterministicRNG:
    """
    Centralized random number generator for test determinism.

    Features:
    - Single source of truth for random state
    - Easy seed control for reproducibility
    - Context manager for temporary seeding
    - Support for NumPy, Python random, and PyTorch
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the RNG with optional seed.

        Args:
            seed: Random seed for reproducibility. If None, uses random initialization.
        """
        self._seed = seed
        self._numpy_rng: Optional[np.random.Generator] = None
        self._initialized = False

        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for all RNG backends.

        Args:
            seed: Random seed value
        """
        self._seed = seed

        np.random.seed(seed)

        random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self._numpy_rng = np.random.default_rng(seed)
        self._initialized = True

    def get_numpy_rng(self) -> np.random.Generator:
        """
        Get the NumPy random generator.

        Returns:
            NumPy Generator instance
        """
        if not self._initialized:
            self.set_seed(self._seed or 42)
        return self._numpy_rng

    def randn(self, *shape: int, dtype: type = np.float64) -> np.ndarray:
        """
        Generate standard normal random numbers.

        Args:
            *shape: Shape of output array
            dtype: Data type for output

        Returns:
            Random array from standard normal distribution
        """
        rng = self.get_numpy_rng()
        return rng.standard_normal(size=shape).astype(dtype)

    def rand(self, *shape: int, dtype: type = np.float64) -> np.ndarray:
        """
        Generate uniform random numbers in [0, 1).

        Args:
            *shape: Shape of output array
            dtype: Data type for output

        Returns:
            Random array from uniform distribution
        """
        rng = self.get_numpy_rng()
        return rng.random(size=shape).astype(dtype)

    def randint(
        self, low: int, high: Optional[int] = None, size: Optional[Union[int, tuple]] = None
    ) -> Union[int, np.ndarray]:
        """
        Generate random integers.

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
        a: Union[int, np.ndarray],
        size: Optional[Union[int, tuple]] = None,
        replace: bool = True,
        p: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate random samples from array.

        Args:
            a: Array to sample from, or int for range(a)
            size: Output shape
            replace: Whether to sample with replacement
            p: Probabilities for each element

        Returns:
            Random sample(s)
        """
        rng = self.get_numpy_rng()
        return rng.choice(a, size=size, replace=replace, p=p)

    def shuffle(self, array: np.ndarray) -> np.ndarray:
        """
        Shuffle array in-place.

        Args:
            array: Array to shuffle

        Returns:
            Shuffled array (same object as input)
        """
        rng = self.get_numpy_rng()
        rng.shuffle(array)
        return array

    def permutation(self, x: Union[int, np.ndarray]) -> np.ndarray:
        """
        Generate random permutation.

        Args:
            x: Array to permute, or int for range(x)

        Returns:
            Permuted array
        """
        rng = self.get_numpy_rng()
        return rng.permutation(x)

    @contextmanager
    def temporary_seed(self, seed: int):
        """
        Context manager for temporary seed override.

        Example:
            with rng.temporary_seed(123):
                data = rng.randn(100, 10)

        Args:
            seed: Temporary seed value
        """
        old_seed = self._seed
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()
        old_torch_state = torch.get_rng_state()

        self.set_seed(seed)

        try:
            yield self
        finally:
            self._seed = old_seed
            np.random.set_state(old_numpy_state)
            random.setstate(old_random_state)
            torch.set_rng_state(old_torch_state)
            if self._seed is not None:
                self._numpy_rng = np.random.default_rng(self._seed)

    def get_seed(self) -> Optional[int]:
        """
        Get the current seed value.

        Returns:
            Current seed or None if not set
        """
        return self._seed

    @staticmethod
    def make_deterministic(seed: int = 42) -> None:
        """
        Make all random operations deterministic across the entire environment.

        Args:
            seed: Seed value for determinism
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


_global_rng: Optional[DeterministicRNG] = None


def get_global_rng() -> DeterministicRNG:
    """
    Get the global RNG instance.

    Returns:
        Global DeterministicRNG instance
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = DeterministicRNG(seed=42)
    return _global_rng


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed.

    Args:
        seed: Random seed value
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = DeterministicRNG(seed=seed)
    else:
        _global_rng.set_seed(seed)


def reset_global_rng() -> None:
    """Reset the global RNG to uninitialized state."""
    global _global_rng
    _global_rng = None
