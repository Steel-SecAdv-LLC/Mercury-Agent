# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Memory-Efficient Feature Caching.

This module provides memory-efficient caching strategies for feature vectors
in anomaly detection pipelines.

Key features:
- Sparse feature representation for high-dimensional detectors
- Feature quantization (INT8/FP16) for reduced memory footprint
- Incremental feature computation (only update changed features)
- LRU cache with memory-aware eviction
- Feature selection using mutual information

These optimizations enable processing of larger datasets and more detectors
while maintaining reasonable memory usage, critical for humanitarian
applications like real-time crisis monitoring and pandemic surveillance.

Memory savings estimates:
- INT8 quantization: 4x reduction vs FP32
- FP16 quantization: 2x reduction vs FP32
- Sparse representation: Variable, depends on sparsity
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np

from omni_mercury_engine._compat import HAS_TORCH

# torch is an optional [ml] dependency. Import it only when available (or for
# type-checking) so the pure-numpy feature-selection helpers in this module —
# compute_feature_importance / select_top_features — and the numpy cache path
# are importable and testable in the torch-free core lane.
if TYPE_CHECKING or HAS_TORCH:
    import torch


def _is_tensor(value: object) -> TypeGuard[torch.Tensor]:
    """Narrowing tensor check that is safe when torch is not installed.

    A plain ``HAS_TORCH and isinstance(x, torch.Tensor)`` is runtime-safe (it
    short-circuits before evaluating ``torch.Tensor``) but does not let mypy
    narrow the ``else`` branch to ``ndarray`` — leaving spurious union-attr
    errors on ``.copy()`` / ``.tobytes()``. A ``TypeGuard`` restores the
    narrowing while keeping the short-circuit.
    """
    return HAS_TORCH and isinstance(value, torch.Tensor)


class QuantizationType(Enum):
    """Feature quantization types."""

    NONE = "none"
    INT8 = "int8"
    FP16 = "fp16"
    DYNAMIC = "dynamic"


@dataclass
class CacheConfig:
    """Configuration for feature cache.

    Attributes:
        max_size: Maximum number of entries in cache
        max_memory_mb: Maximum memory usage in MB
        quantization: Quantization type for stored features
        enable_sparse: Enable sparse representation for high-dimensional features
        sparsity_threshold: Threshold below which values are considered zero
        ttl_seconds: Time-to-live for cache entries (0 = no expiration)
    """

    max_size: int = 1000
    max_memory_mb: float = 512.0
    quantization: QuantizationType = QuantizationType.NONE
    enable_sparse: bool = False
    sparsity_threshold: float = 1e-6
    ttl_seconds: float = 0.0
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    data: np.ndarray[Any, Any] | torch.Tensor
    original_dtype: np.dtype[Any] | torch.dtype
    original_shape: tuple[int, ...]
    is_sparse: bool = False
    sparse_indices: np.ndarray[Any, Any] | None = None
    memory_bytes: int = 0
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    # Affine INT8 dequantization parameters (original = q * scale + zero_point).
    # Set only when the stored ``data`` is uint8 (INT8 / DYNAMIC-int8 path); the
    # reconstruction is lossy without them, so they must ride with the entry.
    quant_scale: float | None = None
    quant_zero_point: float | None = None


class MemoryEfficientFeatureCache:
    """Memory-efficient LRU cache for feature vectors.

    Provides automatic quantization, sparse representation, and memory-aware
    eviction to optimize memory usage while maintaining fast access.

    Example:
        >>> cache = MemoryEfficientFeatureCache(CacheConfig(max_memory_mb=256))
        >>> cache.put("detector1", features)
        >>> cached_features = cache.get("detector1")
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize feature cache.

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._total_memory_bytes = 0
        self._hit_count = 0
        self._miss_count = 0

    def put(
        self,
        key: str,
        data: np.ndarray[Any, Any] | torch.Tensor,
        force: bool = False,
    ) -> bool:
        """Store feature data in cache.

        Args:
            key: Cache key
            data: Feature data to store
            force: Force storage even if memory limit exceeded

        Returns:
            True if stored successfully, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)

            processed_data, is_sparse, sparse_indices, quant_scale, quant_zero_point = (
                self._process_data(data)
            )

            memory_bytes = self._estimate_memory(processed_data)

            if not force:
                while (
                    self._total_memory_bytes + memory_bytes
                    > self.config.max_memory_mb * 1024 * 1024
                    and self._cache
                ):
                    self._evict_oldest()

                if len(self._cache) >= self.config.max_size:
                    self._evict_oldest()

            import time

            now = time.time()

            # Both np.ndarray and torch.Tensor expose ``.dtype`` / ``.shape``.
            original_dtype = data.dtype
            original_shape = tuple(data.shape)

            entry = CacheEntry(
                key=key,
                data=processed_data,
                original_dtype=original_dtype,
                original_shape=original_shape,
                is_sparse=is_sparse,
                sparse_indices=sparse_indices,
                memory_bytes=memory_bytes,
                access_count=0,
                created_at=now,
                last_accessed=now,
                quant_scale=quant_scale,
                quant_zero_point=quant_zero_point,
            )

            self._cache[key] = entry
            self._total_memory_bytes += memory_bytes

            return True

    def get(self, key: str) -> np.ndarray[Any, Any] | torch.Tensor | None:
        """Retrieve feature data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found
        """
        with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None

            entry = self._cache[key]

            if self.config.ttl_seconds > 0:
                import time

                if time.time() - entry.created_at > self.config.ttl_seconds:
                    self._remove_entry(key)
                    self._miss_count += 1
                    return None

            self._cache.move_to_end(key)

            import time

            entry.last_accessed = time.time()
            entry.access_count += 1
            self._hit_count += 1

            return self._reconstruct_data(entry)

    def _process_data(
        self, data: np.ndarray[Any, Any] | torch.Tensor
    ) -> tuple[np.ndarray[Any, Any], bool, Any, float | None, float | None]:
        """Process data for storage with quantization and sparsification.

        Args:
            data: Input data

        Returns:
            Tuple of ``(processed_data, is_sparse, sparse_indices, quant_scale,
            quant_zero_point)``. The two quant params are non-None only on the
            INT8 (or DYNAMIC-int8) path and are required to reconstruct the
            original magnitudes.
        """
        if _is_tensor(data):
            np_data = data.detach().cpu().numpy()
        else:
            # ndarray input: copy so we never mutate the caller's array.
            np_data = np.asarray(data).copy()

        is_sparse = False
        sparse_indices = None

        if self.config.enable_sparse:
            sparsity = np.mean(np.abs(np_data) < self.config.sparsity_threshold)
            if sparsity > 0.5:
                is_sparse = True
                sparse_indices = np.where(np.abs(np_data) >= self.config.sparsity_threshold)
                np_data = np_data[sparse_indices]

        quant_scale: float | None = None
        quant_zero_point: float | None = None
        if self.config.quantization == QuantizationType.INT8:
            np_data, quant_scale, quant_zero_point = self._quantize_int8(np_data)
        elif self.config.quantization == QuantizationType.FP16:
            np_data = np_data.astype(np.float16)
        elif self.config.quantization == QuantizationType.DYNAMIC:
            if np_data.size > 10000:
                np_data, quant_scale, quant_zero_point = self._quantize_int8(np_data)
            elif np_data.size > 1000:
                np_data = np_data.astype(np.float16)

        return np_data, is_sparse, sparse_indices, quant_scale, quant_zero_point

    def _quantize_int8(
        self, data: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], float, float]:
        """Affine-quantize data to uint8, returning the dequant parameters.

        Maps ``[min, max] -> [0, 255]`` as ``q = round((x - min) / scale)`` with
        ``scale = (max - min) / 255``. The inverse (applied in
        :meth:`_reconstruct_data`) is ``x = q * scale + min``, so both ``scale``
        and the zero point (``min``) must be returned and stored — reconstructing
        with only ``/ 255`` (the previous behaviour) corrupted every cached
        value to roughly ``[0, 1]`` regardless of its true magnitude.

        Args:
            data: Input data.

        Returns:
            ``(quantized_uint8, scale, zero_point)``.
        """
        min_val = float(data.min())
        max_val = float(data.max())
        # Degenerate (constant) array: scale 1.0 so every value maps to 0 and
        # reconstructs exactly to min_val.
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0

        quantized = np.round((data - min_val) / scale).astype(np.uint8)

        return np.asarray(quantized), scale, min_val

    def _reconstruct_data(self, entry: CacheEntry) -> np.ndarray[Any, Any]:
        """Reconstruct original data from cache entry.

        Args:
            entry: Cache entry

        Returns:
            Reconstructed data
        """
        data = entry.data

        if (
            isinstance(data, np.ndarray)
            and data.dtype == np.uint8
            and entry.quant_scale is not None
            and entry.quant_zero_point is not None
        ):
            # Invert the affine quantization: x = q * scale + zero_point.
            # (The previous ``/ 255`` restored neither scale nor zero point and
            # corrupted every dequantized value to ~[0, 1].)
            data = data.astype(np.float32) * entry.quant_scale + entry.quant_zero_point

        if self.config.quantization == QuantizationType.FP16:
            if isinstance(data, np.ndarray):
                data = data.astype(np.float32)

        if entry.is_sparse and entry.sparse_indices is not None:
            full_data = np.zeros(entry.original_shape, dtype=np.float32)
            if isinstance(data, np.ndarray):
                full_data[entry.sparse_indices] = data
            data = full_data

        if isinstance(data, np.ndarray):
            return data
        # Convert torch.Tensor to numpy array if needed
        return np.asarray(data)

    def _estimate_memory(self, data: np.ndarray[Any, Any]) -> int:
        """Estimate memory usage of data.

        Args:
            data: Data to estimate

        Returns:
            Estimated memory in bytes
        """
        return int(data.nbytes)

    def _evict_oldest(self) -> None:
        """Evict oldest entry from cache."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache.

        Args:
            key: Key to remove
        """
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_memory_bytes -= entry.memory_bytes

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_memory_bytes = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.config.max_size,
                "memory_mb": self._total_memory_bytes / (1024 * 1024),
                "max_memory_mb": self.config.max_memory_mb,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "quantization": self.config.quantization.value,
                "sparse_enabled": self.config.enable_sparse,
            }

    def contains(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise
        """
        with self._lock:
            return key in self._cache


class IncrementalFeatureComputer:
    """Incremental feature computation for efficient updates.

    Only recomputes features that have changed, caching intermediate results for faster subsequent
    computations.
    """

    def __init__(self, cache: MemoryEfficientFeatureCache | None = None) -> None:
        """Initialize incremental feature computer.

        Args:
            cache: Optional feature cache to use
        """
        self.cache = cache or MemoryEfficientFeatureCache()
        self._data_hashes: dict[str, str] = {}

    def compute_hash(self, data: np.ndarray[Any, Any] | torch.Tensor) -> str:
        """Compute hash of data for change detection.

        Args:
            data: Data to hash

        Returns:
            Hash string
        """
        array = data.detach().cpu().numpy() if _is_tensor(data) else np.asarray(data)

        # Use SHA3-256 for AMA Cryptography alignment (non-cryptographic use for cache keys)
        return hashlib.sha3_256(array.tobytes()).hexdigest()

    def needs_update(self, key: str, data: np.ndarray[Any, Any] | torch.Tensor) -> bool:
        """Check if features need to be recomputed.

        Args:
            key: Feature key
            data: Input data

        Returns:
            True if features need update, False otherwise
        """
        current_hash = self.compute_hash(data)
        stored_hash = self._data_hashes.get(key)

        return stored_hash != current_hash

    def update_features(
        self,
        key: str,
        data: np.ndarray[Any, Any] | torch.Tensor,
        features: np.ndarray[Any, Any] | torch.Tensor,
    ) -> None:
        """Update cached features.

        Args:
            key: Feature key
            data: Input data (for hash computation)
            features: Computed features to cache
        """
        self._data_hashes[key] = self.compute_hash(data)
        self.cache.put(key, features)

    def get_features(self, key: str) -> np.ndarray[Any, Any] | torch.Tensor | None:
        """Get cached features.

        Args:
            key: Feature key

        Returns:
            Cached features or None if not found
        """
        return self.cache.get(key)


def compute_feature_importance(
    features: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any] | None = None,
    method: str = "variance",
) -> np.ndarray[Any, Any]:
    """Compute feature importance for feature selection.

    Args:
        features: Feature matrix [n_samples, n_features]
        labels: Optional labels for supervised methods
        method: Importance method ("variance", "mutual_info", "correlation")

    Returns:
        Importance scores for each feature
    """
    if method == "variance":
        return np.asarray(np.var(features, axis=0))

    elif method == "correlation" and labels is not None:
        correlations = np.zeros(features.shape[1])
        for i in range(features.shape[1]):
            if np.std(features[:, i]) > 0:
                correlations[i] = np.abs(np.corrcoef(features[:, i], labels)[0, 1])
        return correlations

    elif method == "mutual_info" and labels is not None:
        mi_scores = np.zeros(features.shape[1])
        for i in range(features.shape[1]):
            hist_2d, _, _ = np.histogram2d(features[:, i], labels, bins=10)
            pxy = hist_2d / hist_2d.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)

            px_py = np.outer(px, py)
            nonzero = pxy > 0
            mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
            mi_scores[i] = mi

        return mi_scores

    return np.asarray(np.var(features, axis=0))


def select_top_features(
    features: np.ndarray[Any, Any],
    importance: np.ndarray[Any, Any],
    k: int | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Select top features based on importance scores.

    Args:
        features: Feature matrix [n_samples, n_features]
        importance: Importance scores for each feature
        k: Number of top features to select
        threshold: Minimum importance threshold

    Returns:
        Tuple of (selected_features, selected_indices)
    """
    if k is not None:
        k = min(max(k, 0), len(importance))
        # ``np.argsort(...)[-k:]`` returns the WHOLE array when k == 0 (``[-0:]``
        # is ``[0:]``), so select-zero-features silently returned every feature.
        # Guard k == 0 explicitly to select nothing.
        if k == 0:
            indices = np.array([], dtype=np.intp)
        else:
            indices = np.argsort(importance)[-k:]
    elif threshold is not None:
        indices = np.where(importance >= threshold)[0]
    else:
        indices = np.arange(len(importance))

    return features[:, indices], indices
