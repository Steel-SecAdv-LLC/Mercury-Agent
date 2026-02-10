"""
Mercury Agent ♱
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

from __future__ import annotations

"""
Efficiency Optimizations Module

Provides performance optimizations for Mercury Agent:
- Joblib parallelization for benchmark loops
- torch.compile() integration for 2x fusion network speedup
- LRU cache management with memory limits (128MB cap)
- Multi-GPU DDP scaling
- Memory-efficient operations

Runtime target: ~50min benchmarks → ~25min with optimizations.
"""

import gc
import logging
import os
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class OptimizationConfig:
    """Configuration for efficiency optimizations."""

    # Parallelization
    enable_joblib: bool = True
    n_jobs: int = -1  # -1 for all cores
    joblib_backend: str = "loky"
    prefer_threads: bool = False

    # Torch optimizations
    enable_torch_compile: bool = True
    torch_compile_mode: str = "reduce-overhead"  # default, reduce-overhead, max-autotune
    torch_compile_fullgraph: bool = False

    # Memory management
    cache_max_size_mb: int = 128
    cache_max_entries: int = 128
    enable_memory_tracking: bool = True
    memory_threshold_mb: float = 2048.0
    gc_threshold: float = 0.8  # Trigger GC at 80% of threshold

    # Multi-GPU
    enable_ddp: bool = False
    ddp_backend: str = "nccl"
    world_size: int = 1
    local_rank: int = 0


class MemoryEfficientCache:
    """
    LRU cache with memory limits.

    Unlike functools.lru_cache, this implementation:
    - Tracks total memory usage
    - Evicts when memory limit is reached (128MB default)
    - Supports size estimation for numpy/torch objects
    """

    def __init__(
        self,
        max_size_mb: float = 128.0,
        max_entries: int = 128,
    ):
        """
        Initialize memory-efficient cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cache entries
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.cache: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of an object."""
        if hasattr(obj, "nbytes"):
            return int(obj.nbytes)
        elif hasattr(obj, "element_size") and hasattr(obj, "numel"):
            # PyTorch tensor
            return int(obj.element_size() * obj.numel())
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(x) for x in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
        else:
            return sys.getsizeof(obj)

    def get(self, key: str) -> Any | None:
        """Get item from cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key][0]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache with automatic eviction."""
        size = self._estimate_size(value)

        # Evict if necessary
        while (
            self.current_size + size > self.max_size_bytes or len(self.cache) >= self.max_entries
        ) and self.cache:
            oldest_key, (_, oldest_size) = self.cache.popitem(last=False)
            self.current_size -= oldest_size

        # Add new entry
        self.cache[key] = (value, size)
        self.current_size += size

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.current_size = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "entries": len(self.cache),
            "size_mb": self.current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
        }


def memory_efficient_lru_cache(
    max_size_mb: float = 128.0,
    max_entries: int = 128,
) -> Any:
    """
    Decorator for memory-efficient LRU caching.

    Args:
        max_size_mb: Maximum cache size in MB
        max_entries: Maximum number of entries

    Returns:
        Decorated function with memory-aware caching
    """
    cache = MemoryEfficientCache(max_size_mb, max_entries)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key
            key = str((args, tuple(sorted(kwargs.items()))))

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = lambda: cache.stats  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


class MemoryManager:
    """
    Memory manager for preventing OOM errors.

    Monitors memory usage and triggers garbage collection
    when approaching limits.
    """

    def __init__(
        self,
        threshold_mb: float = 2048.0,
        gc_threshold: float = 0.8,
    ):
        """
        Initialize memory manager.

        Args:
            threshold_mb: Memory threshold in MB
            gc_threshold: GC trigger ratio (0-1)
        """
        self.threshold_bytes = int(threshold_mb * 1024 * 1024)
        self.gc_threshold = gc_threshold
        self._torch_available = False

        try:
            import importlib.util

            self._torch_available = importlib.util.find_spec("torch") is not None
        except Exception:
            pass  # torch availability check failed, assume not available

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()

        usage = {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024),
        }

        if self._torch_available:
            import torch

            if torch.cuda.is_available():
                usage["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                usage["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)

        return usage

    def check_and_cleanup(self) -> bool:
        """
        Check memory and cleanup if needed.

        Returns:
            True if cleanup was performed
        """
        try:
            import psutil

            mem_usage = psutil.Process().memory_info().rss

            if mem_usage > self.threshold_bytes * self.gc_threshold:
                logger.info(
                    f"Memory usage {mem_usage / (1024 * 1024):.1f}MB exceeds threshold, running GC"
                )
                gc.collect()

                if self._torch_available:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                return True
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

        return False


def compile_model(
    model: Any,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
) -> Any:
    """
    Apply torch.compile() optimization to a model.

    Provides ~2x speedup for fusion networks on compatible hardware.

    Args:
        model: PyTorch model to compile
        mode: Compilation mode (default, reduce-overhead, max-autotune)
        fullgraph: Whether to compile as fullgraph

    Returns:
        Compiled model (or original if compilation fails)
    """
    try:
        import torch

        if not hasattr(torch, "compile"):
            logger.info("torch.compile not available (requires PyTorch 2.0+)")
            return model

        # Check if model is already compiled
        if hasattr(model, "_orig_mod"):
            logger.debug("Model already compiled")
            return model

        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        logger.info(f"Model compiled with mode={mode}")
        return compiled

    except Exception as e:
        logger.warning(f"torch.compile failed, using original model: {e}")
        return model


class ParallelExecutor:
    """
    Parallel executor using joblib for benchmark loops.

    Provides easy parallelization of independent operations
    like dataset processing and multi-detector evaluation.
    """

    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = "loky",
        prefer: str | None = None,
    ):
        """
        Initialize parallel executor.

        Args:
            n_jobs: Number of jobs (-1 for all cores)
            backend: Joblib backend (loky, threading, multiprocessing)
            prefer: Preference hint (threads, processes)
        """
        self.n_jobs = n_jobs
        self.backend = backend
        self.prefer = prefer
        self._joblib_available = False

        try:
            import importlib.util

            self._joblib_available = importlib.util.find_spec("joblib") is not None
        except Exception:
            logger.warning("joblib not installed, parallel execution disabled")

    def map(
        self,
        func: Any,
        items: list[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """
        Map function over items in parallel.

        Args:
            func: Function to apply
            items: List of items to process
            **kwargs: Additional arguments to pass to func

        Returns:
            List of results
        """
        if not self._joblib_available or self.n_jobs == 1:
            return [func(item, **kwargs) for item in items]

        from joblib import Parallel, delayed

        return list(
            Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)(
                delayed(func)(item, **kwargs) for item in items
            )
        )

    def starmap(
        self,
        func: Any,
        args_list: list[tuple[Any, ...]],
    ) -> list[Any]:
        """
        Map function over argument tuples in parallel.

        Args:
            func: Function to apply
            args_list: List of argument tuples

        Returns:
            List of results
        """
        if not self._joblib_available or self.n_jobs == 1:
            return [func(*args) for args in args_list]

        from joblib import Parallel, delayed

        return list(
            Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)(
                delayed(func)(*args) for args in args_list
            )
        )


class DDPScaler:
    """
    Distributed Data Parallel scaler for multi-GPU training.

    Provides easy setup for DDP training across multiple GPUs.
    """

    def __init__(
        self,
        backend: str = "nccl",
        world_size: int | None = None,
        local_rank: int | None = None,
    ):
        """
        Initialize DDP scaler.

        Args:
            backend: DDP backend (nccl, gloo)
            world_size: Number of processes
            local_rank: Local process rank
        """
        self.backend = backend
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = local_rank or int(os.environ.get("LOCAL_RANK", "0"))
        self.is_initialized = False

    def setup(self) -> bool:
        """
        Setup DDP environment.

        Returns:
            True if setup successful
        """
        if self.world_size <= 1:
            logger.info("Single GPU/CPU mode, DDP not needed")
            return False

        try:
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    world_size=self.world_size,
                    rank=self.local_rank,
                )
                self.is_initialized = True
                logger.info(f"DDP initialized: rank {self.local_rank}/{self.world_size}")
            return True

        except Exception as e:
            logger.error(f"DDP setup failed: {e}")
            return False

    def wrap_model(self, model: Any) -> Any:
        """
        Wrap model with DDP.

        Args:
            model: PyTorch model

        Returns:
            DDP-wrapped model
        """
        if not self.is_initialized:
            return model

        try:
            import torch
            from torch.nn.parallel import DistributedDataParallel as DDP

            device = torch.device(f"cuda:{self.local_rank}")
            model = model.to(device)
            return DDP(model, device_ids=[self.local_rank])

        except Exception as e:
            logger.error(f"DDP model wrap failed: {e}")
            return model

    def cleanup(self) -> None:
        """Cleanup DDP resources."""
        if self.is_initialized:
            try:
                import torch.distributed as dist

                dist.destroy_process_group()
                self.is_initialized = False
            except Exception as e:
                logger.debug("DDP cleanup: process group may not exist or already destroyed: %s", e)
                self.is_initialized = False


# Alias for backward compatibility and intuitive naming
DDPManager = DDPScaler


def estimate_batch_size(
    sample_size_bytes: int,
    available_memory_mb: float = 2048.0,
    memory_fraction: float = 0.7,
    min_batch: int = 1,
    max_batch: int = 4096,
) -> int:
    """
    Estimate optimal batch size based on available memory.

    Computes how many samples can fit in memory given per-sample size,
    reserving headroom for model parameters and intermediate activations.

    Args:
        sample_size_bytes: Size of a single sample in bytes
        available_memory_mb: Total available memory in MB
        memory_fraction: Fraction of memory to use for batches (rest for model, etc.)
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Estimated batch size clamped to [min_batch, max_batch]
    """
    if sample_size_bytes <= 0:
        logger.warning(
            "Invalid sample_size_bytes (%d), returning default batch size", sample_size_bytes
        )
        return min_batch

    try:
        usable_bytes = available_memory_mb * 1024 * 1024 * memory_fraction
        estimated = int(usable_bytes / sample_size_bytes)
        return max(min_batch, min(max_batch, estimated))
    except Exception as e:
        logger.warning("Batch size estimation failed: %s, returning default", e)
        return min_batch


@dataclass
class OptimizationResult:
    """Result of optimization application."""

    original_time_ms: float
    optimized_time_ms: float
    speedup: float
    optimizations_applied: list[str]
    memory_saved_mb: float = 0.0


def apply_all_optimizations(
    model: Any | None = None,
    config: OptimizationConfig | None = None,
) -> dict[str, Any]:
    """
    Apply all available optimizations.

    Args:
        model: Optional model to optimize
        config: Optimization configuration

    Returns:
        Dictionary with optimization components
    """
    config = config or OptimizationConfig()
    components: dict[str, Any] = {}

    # Memory manager
    components["memory_manager"] = MemoryManager(
        threshold_mb=config.memory_threshold_mb,
        gc_threshold=config.gc_threshold,
    )

    # Parallel executor
    if config.enable_joblib:
        components["parallel_executor"] = ParallelExecutor(
            n_jobs=config.n_jobs,
            backend=config.joblib_backend,
            prefer="threads" if config.prefer_threads else None,
        )

    # DDP scaler
    if config.enable_ddp:
        components["ddp_scaler"] = DDPScaler(
            backend=config.ddp_backend,
            world_size=config.world_size,
            local_rank=config.local_rank,
        )

    # Compiled model
    if model is not None and config.enable_torch_compile:
        components["compiled_model"] = compile_model(
            model,
            mode=config.torch_compile_mode,
            fullgraph=config.torch_compile_fullgraph,
        )

    # Cache
    components["cache"] = MemoryEfficientCache(
        max_size_mb=config.cache_max_size_mb,
        max_entries=config.cache_max_entries,
    )

    return components


# Convenience functions
def parallel_map(
    func: Any,
    items: list[Any],
    n_jobs: int = -1,
    **kwargs: Any,
) -> list[Any]:
    """
    Convenience function for parallel mapping.

    Args:
        func: Function to apply
        items: Items to process
        n_jobs: Number of parallel jobs
        **kwargs: Additional function arguments

    Returns:
        List of results
    """
    executor = ParallelExecutor(n_jobs=n_jobs)
    return executor.map(func, items, **kwargs)


def get_optimal_batch_size(
    model: Any,
    input_shape: tuple[int, ...],
    max_memory_fraction: float = 0.8,
) -> int:
    """
    Estimate optimal batch size based on available memory.

    Args:
        model: PyTorch model
        input_shape: Shape of single input (excluding batch)
        max_memory_fraction: Maximum fraction of GPU memory to use

    Returns:
        Recommended batch size
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 32  # Default for CPU

        # Get available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available = int(total_memory * max_memory_fraction)

        # Estimate model memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        # Estimate input memory (float32)
        input_memory = int(np.prod(input_shape)) * 4

        # Estimate gradient memory (roughly 2x parameters)
        gradient_memory = param_memory * 2

        # Available for batch
        batch_memory = available - param_memory - gradient_memory

        # Estimate batch size
        batch_size = max(1, batch_memory // (input_memory * 3))  # 3x for activations

        return int(min(batch_size, 256))  # Cap at 256

    except Exception as e:
        logger.debug("Batch size estimation failed, using default: %s", e)
        return 32  # Default fallback
