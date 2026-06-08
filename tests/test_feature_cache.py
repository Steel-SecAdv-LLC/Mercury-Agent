# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for memory-efficient feature caching."""

import pytest

pytest.importorskip("torch")

import time

import numpy as np
import torch

from omni_mercury_engine.utils.feature_cache import (
    CacheConfig,
    CacheEntry,
    IncrementalFeatureComputer,
    MemoryEfficientFeatureCache,
    QuantizationType,
    compute_feature_importance,
    select_top_features,
)


class TestQuantizationType:
    """Tests for QuantizationType enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert QuantizationType.NONE.value == "none"
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.FP16.value == "fp16"
        assert QuantizationType.DYNAMIC.value == "dynamic"


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()
        assert config.max_size == 1000
        assert config.max_memory_mb == 512.0
        assert config.quantization == QuantizationType.NONE
        assert config.enable_sparse is False
        assert config.sparsity_threshold == 1e-6
        assert config.ttl_seconds == 0.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CacheConfig(
            max_size=500,
            max_memory_mb=256.0,
            quantization=QuantizationType.INT8,
            enable_sparse=True,
            sparsity_threshold=1e-4,
            ttl_seconds=60.0,
        )
        assert config.max_size == 500
        assert config.max_memory_mb == 256.0
        assert config.quantization == QuantizationType.INT8
        assert config.enable_sparse is True


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self) -> None:
        """Test cache entry creation."""
        data = np.array([1.0, 2.0, 3.0])
        entry = CacheEntry(
            key="test_key",
            data=data,
            original_dtype=data.dtype,
            original_shape=data.shape,
            is_sparse=False,
            sparse_indices=None,
            memory_bytes=data.nbytes,
            access_count=0,
            created_at=time.time(),
            last_accessed=time.time(),
        )
        assert entry.key == "test_key"
        assert entry.is_sparse is False
        assert entry.access_count == 0


class TestMemoryEfficientFeatureCache:
    """Tests for MemoryEfficientFeatureCache class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        cache = MemoryEfficientFeatureCache()
        assert cache.config.max_size == 1000
        assert cache._total_memory_bytes == 0

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = CacheConfig(max_size=100)
        cache = MemoryEfficientFeatureCache(config)
        assert cache.config.max_size == 100

    def test_put_and_get_numpy(self) -> None:
        """Test storing and retrieving numpy array."""
        cache = MemoryEfficientFeatureCache()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cache.put("test_key", data)
        retrieved = cache.get("test_key")
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, data)

    def test_put_and_get_torch(self) -> None:
        """Test storing and retrieving torch tensor."""
        cache = MemoryEfficientFeatureCache()
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cache.put("test_key", data)
        retrieved = cache.get("test_key")
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, data.numpy())

    def test_get_nonexistent_key(self) -> None:
        """Test getting nonexistent key returns None."""
        cache = MemoryEfficientFeatureCache()
        assert cache.get("nonexistent") is None

    def test_overwrite_existing_key(self) -> None:
        """Test overwriting existing key."""
        cache = MemoryEfficientFeatureCache()
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])
        cache.put("test_key", data1)
        cache.put("test_key", data2)
        retrieved = cache.get("test_key")
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, data2)

    def test_contains(self) -> None:
        """Test contains method."""
        cache = MemoryEfficientFeatureCache()
        data = np.array([1.0, 2.0, 3.0])
        cache.put("test_key", data)
        assert cache.contains("test_key") is True
        assert cache.contains("nonexistent") is False

    def test_clear(self) -> None:
        """Test clearing cache."""
        cache = MemoryEfficientFeatureCache()
        cache.put("key1", np.array([1.0, 2.0]))
        cache.put("key2", np.array([3.0, 4.0]))
        cache.clear()
        assert cache.contains("key1") is False
        assert cache.contains("key2") is False
        assert cache._total_memory_bytes == 0

    def test_get_stats(self) -> None:
        """Test getting cache statistics."""
        cache = MemoryEfficientFeatureCache()
        cache.put("key1", np.array([1.0, 2.0, 3.0]))
        cache.get("key1")
        cache.get("nonexistent")
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == 0.5

    def test_eviction_by_size(self) -> None:
        """Test eviction when max size is reached."""
        config = CacheConfig(max_size=2)
        cache = MemoryEfficientFeatureCache(config)
        cache.put("key1", np.array([1.0]))
        cache.put("key2", np.array([2.0]))
        cache.put("key3", np.array([3.0]))
        assert cache.contains("key1") is False
        assert cache.contains("key2") is True
        assert cache.contains("key3") is True

    def test_eviction_by_memory(self) -> None:
        """Test eviction when max memory is reached."""
        config = CacheConfig(max_memory_mb=0.0001)
        cache = MemoryEfficientFeatureCache(config)
        cache.put("key1", np.zeros(100))
        cache.put("key2", np.zeros(100))
        assert cache.get_stats()["size"] <= 2

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        config = CacheConfig(ttl_seconds=0.1)
        cache = MemoryEfficientFeatureCache(config)
        cache.put("key1", np.array([1.0, 2.0, 3.0]))
        assert cache.get("key1") is not None
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_int8_quantization(self) -> None:
        """Test INT8 quantization."""
        config = CacheConfig(quantization=QuantizationType.INT8)
        cache = MemoryEfficientFeatureCache(config)
        data = np.array([0.0, 0.5, 1.0])
        cache.put("key1", data)
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.dtype == np.float32

    def test_fp16_quantization(self) -> None:
        """Test FP16 quantization."""
        config = CacheConfig(quantization=QuantizationType.FP16)
        cache = MemoryEfficientFeatureCache(config)
        data = np.array([1.0, 2.0, 3.0])
        cache.put("key1", data)
        retrieved = cache.get("key1")
        assert retrieved is not None

    def test_sparse_representation(self) -> None:
        """Test sparse representation."""
        config = CacheConfig(enable_sparse=True, sparsity_threshold=0.1)
        cache = MemoryEfficientFeatureCache(config)
        data = np.zeros(100)
        data[10] = 1.0
        data[50] = 2.0
        cache.put("key1", data)
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.shape == data.shape

    def test_force_put(self) -> None:
        """Test force put ignores memory limits."""
        config = CacheConfig(max_memory_mb=0.0001)
        cache = MemoryEfficientFeatureCache(config)
        large_data = np.zeros(10000)
        result = cache.put("key1", large_data, force=True)
        assert result is True


class TestIncrementalFeatureComputer:
    """Tests for IncrementalFeatureComputer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        computer = IncrementalFeatureComputer()
        assert computer.cache is not None

    def test_init_with_cache(self) -> None:
        """Test initialization with custom cache."""
        cache = MemoryEfficientFeatureCache()
        computer = IncrementalFeatureComputer(cache=cache)
        assert computer.cache is cache

    def test_compute_hash_numpy(self) -> None:
        """Test hash computation for numpy array."""
        computer = IncrementalFeatureComputer()
        data = np.array([1.0, 2.0, 3.0])
        hash1 = computer.compute_hash(data)
        hash2 = computer.compute_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_compute_hash_torch(self) -> None:
        """Test hash computation for torch tensor."""
        computer = IncrementalFeatureComputer()
        data = torch.tensor([1.0, 2.0, 3.0])
        hash_val = computer.compute_hash(data)
        assert len(hash_val) == 64

    def test_compute_hash_different_data(self) -> None:
        """Test different data produces different hashes."""
        computer = IncrementalFeatureComputer()
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([1.0, 2.0, 4.0])
        assert computer.compute_hash(data1) != computer.compute_hash(data2)

    def test_needs_update_new_key(self) -> None:
        """Test needs_update for new key."""
        computer = IncrementalFeatureComputer()
        data = np.array([1.0, 2.0, 3.0])
        assert computer.needs_update("key1", data) is True

    def test_needs_update_same_data(self) -> None:
        """Test needs_update with same data."""
        computer = IncrementalFeatureComputer()
        data = np.array([1.0, 2.0, 3.0])
        features = np.array([0.5, 0.5])
        computer.update_features("key1", data, features)
        assert computer.needs_update("key1", data) is False

    def test_needs_update_changed_data(self) -> None:
        """Test needs_update with changed data."""
        computer = IncrementalFeatureComputer()
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([1.0, 2.0, 4.0])
        features = np.array([0.5, 0.5])
        computer.update_features("key1", data1, features)
        assert computer.needs_update("key1", data2) is True

    def test_update_and_get_features(self) -> None:
        """Test updating and getting features."""
        computer = IncrementalFeatureComputer()
        data = np.array([1.0, 2.0, 3.0])
        features = np.array([0.5, 0.5, 0.5])
        computer.update_features("key1", data, features)
        retrieved = computer.get_features("key1")
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, features)

    def test_get_features_nonexistent(self) -> None:
        """Test getting nonexistent features."""
        computer = IncrementalFeatureComputer()
        assert computer.get_features("nonexistent") is None


class TestComputeFeatureImportance:
    """Tests for compute_feature_importance function."""

    def test_variance_method(self) -> None:
        """Test variance-based importance."""
        features = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        importance = compute_feature_importance(features, method="variance")
        assert importance[1] > importance[0]

    def test_correlation_method(self) -> None:
        """Test correlation-based importance."""
        features = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        labels = np.array([1.0, 2.0, 3.0])
        importance = compute_feature_importance(features, labels, method="correlation")
        assert importance[0] > importance[1]

    def test_mutual_info_method(self) -> None:
        """Test mutual information-based importance."""
        features = np.array([[1.0, 0.5], [2.0, 0.5], [3.0, 0.5], [4.0, 0.5]])
        labels = np.array([0, 0, 1, 1])
        importance = compute_feature_importance(features, labels, method="mutual_info")
        assert len(importance) == 2

    def test_default_method(self) -> None:
        """Test default method (variance)."""
        features = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        importance = compute_feature_importance(features)
        assert len(importance) == 2

    def test_unknown_method_fallback(self) -> None:
        """Test unknown method falls back to variance."""
        features = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        importance = compute_feature_importance(features, method="unknown")
        assert len(importance) == 2


class TestSelectTopFeatures:
    """Tests for select_top_features function."""

    def test_select_by_k(self) -> None:
        """Test selecting top k features."""
        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        importance = np.array([0.1, 0.5, 0.3])
        selected, indices = select_top_features(features, importance, k=2)
        assert selected.shape[1] == 2
        assert len(indices) == 2
        assert 1 in indices

    def test_select_by_threshold(self) -> None:
        """Test selecting features by threshold."""
        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        importance = np.array([0.1, 0.5, 0.3])
        selected, indices = select_top_features(features, importance, threshold=0.25)
        assert 1 in indices
        assert 2 in indices

    def test_select_all_features(self) -> None:
        """Test selecting all features when no k or threshold."""
        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        importance = np.array([0.1, 0.5, 0.3])
        selected, indices = select_top_features(features, importance)
        assert selected.shape[1] == 3
        assert len(indices) == 3

    def test_k_larger_than_features(self) -> None:
        """Test k larger than number of features."""
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        importance = np.array([0.1, 0.5])
        selected, indices = select_top_features(features, importance, k=10)
        assert selected.shape[1] == 2
