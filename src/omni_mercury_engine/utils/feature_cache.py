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
Memory-Efficient Feature Caching

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

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch


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
    original_dtype: np.dtype | torch.dtype
    original_shape: tuple[int, ...]
    is_sparse: bool = False
    sparse_indices: np.ndarray[Any, Any] | None = None
    memory_bytes: int = 0
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0


class MemoryEfficientFeatureCache:
    """
    Memory-efficient LRU cache for feature vectors.

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

            processed_data, is_sparse, sparse_indices = self._process_data(data)

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

            original_dtype = data.dtype if isinstance(data, np.ndarray) else data.dtype
            original_shape = data.shape

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
    ) -> tuple[np.ndarray[Any, Any], bool, Any]:
        """Process data for storage with quantization and sparsification.

        Args:
            data: Input data

        Returns:
            Tuple of (processed_data, is_sparse, sparse_indices)
        """
        np_data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data.copy()

        is_sparse = False
        sparse_indices = None

        if self.config.enable_sparse:
            sparsity = np.mean(np.abs(np_data) < self.config.sparsity_threshold)
            if sparsity > 0.5:
                is_sparse = True
                sparse_indices = np.where(np.abs(np_data) >= self.config.sparsity_threshold)
                np_data = np_data[sparse_indices]

        if self.config.quantization == QuantizationType.INT8:
            np_data = self._quantize_int8(np_data)
        elif self.config.quantization == QuantizationType.FP16:
            np_data = np_data.astype(np.float16)
        elif self.config.quantization == QuantizationType.DYNAMIC:
            if np_data.size > 10000:
                np_data = self._quantize_int8(np_data)
            elif np_data.size > 1000:
                np_data = np_data.astype(np.float16)

        return np_data, is_sparse, sparse_indices

    def _quantize_int8(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Quantize data to INT8.

        Args:
            data: Input data

        Returns:
            Quantized data
        """
        min_val = data.min()
        max_val = data.max()
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0

        quantized = ((data - min_val) / scale).astype(np.uint8)

        return np.asarray(quantized)

    def _reconstruct_data(self, entry: CacheEntry) -> np.ndarray[Any, Any]:
        """Reconstruct original data from cache entry.

        Args:
            entry: Cache entry

        Returns:
            Reconstructed data
        """
        data = entry.data

        if self.config.quantization in (QuantizationType.INT8, QuantizationType.DYNAMIC):
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0

        if self.config.quantization == QuantizationType.FP16:
            data = data.astype(np.float32)

        if entry.is_sparse and entry.sparse_indices is not None:
            full_data = np.zeros(entry.original_shape, dtype=np.float32)
            full_data[entry.sparse_indices] = data
            data = full_data

        return data

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
    """
    Incremental feature computation for efficient updates.

    Only recomputes features that have changed, caching intermediate
    results for faster subsequent computations.
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
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Use SHA-256 instead of MD5 for better security (non-cryptographic use for cache keys)
        return hashlib.sha256(data.tobytes()).hexdigest()

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
        k = min(k, len(importance))
        indices = np.argsort(importance)[-k:]
    elif threshold is not None:
        indices = np.where(importance >= threshold)[0]
    else:
        indices = np.arange(len(importance))

    return features[:, indices], indices
