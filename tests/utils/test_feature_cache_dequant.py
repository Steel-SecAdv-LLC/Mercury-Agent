# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Core-lane tests for feature_cache correctness fixes (no torch required).

These pin two real defects and prove the module's pure-numpy surface works
without the optional ``torch`` extra (the module now imports torch lazily):

1. INT8 / DYNAMIC-int8 dequantization restored magnitudes correctly. The old
   reconstruction divided the uint8 codes by 255 without restoring the scale
   or zero point, corrupting every dequantized value to roughly [0, 1]
   regardless of the true range. The prior test only checked dtype, so the
   corruption went unnoticed; these assert value fidelity.
2. ``select_top_features(k=0)`` selected ZERO features (it previously returned
   the whole matrix because ``np.argsort(...)[-0:]`` is ``[0:]``).
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.utils.feature_cache import (
    CacheConfig,
    MemoryEfficientFeatureCache,
    QuantizationType,
    compute_feature_importance,
    select_top_features,
)


class TestInt8DequantFidelity:
    def test_int8_round_trip_preserves_magnitude(self) -> None:
        # A wide-range array is where the old /255 bug was catastrophic:
        # values in [-50, 50] came back as ~[0, 1].
        rng = np.random.default_rng(0)
        original = rng.uniform(-50.0, 50.0, size=(4, 8)).astype(np.float32)
        cache = MemoryEfficientFeatureCache(CacheConfig(quantization=QuantizationType.INT8))
        cache.put("k", original)
        recon = cache.get("k")
        assert recon is not None
        # Reconstruction must land within one quantization step of the original,
        # NOT be squashed into [0, 1].
        step = (original.max() - original.min()) / 255.0
        assert np.max(np.abs(recon - original)) <= step + 1e-4
        assert recon.min() < -1.0 and recon.max() > 1.0, (
            "dequantized values were squashed toward [0,1] — the scale/zero-point "
            "were not restored"
        )

    def test_int8_constant_array_round_trips_exactly(self) -> None:
        cache = MemoryEfficientFeatureCache(CacheConfig(quantization=QuantizationType.INT8))
        original = np.full((3, 3), 7.0, dtype=np.float32)
        cache.put("c", original)
        recon = cache.get("c")
        assert recon is not None
        np.testing.assert_allclose(recon, original, atol=1e-5)

    def test_dynamic_int8_large_array_round_trips(self) -> None:
        # DYNAMIC quantization routes arrays > 10k elements through int8.
        rng = np.random.default_rng(1)
        original = rng.uniform(-10.0, 10.0, size=(200, 60)).astype(np.float32)
        assert original.size > 10000
        cache = MemoryEfficientFeatureCache(CacheConfig(quantization=QuantizationType.DYNAMIC))
        cache.put("big", original)
        recon = cache.get("big")
        assert recon is not None
        step = (original.max() - original.min()) / 255.0
        assert np.max(np.abs(recon - original)) <= step + 1e-4


class TestSelectTopFeatures:
    def test_k_zero_selects_nothing(self) -> None:
        features = np.random.default_rng(0).random((5, 6))
        importance = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.5])
        selected, indices = select_top_features(features, importance, k=0)
        assert selected.shape == (5, 0)
        assert indices.size == 0

    def test_k_selects_correct_top(self) -> None:
        features = np.random.default_rng(0).random((5, 6))
        importance = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.5])
        _, indices = select_top_features(features, importance, k=2)
        assert set(indices.tolist()) == {1, 3}

    def test_k_larger_than_n_selects_all(self) -> None:
        features = np.random.default_rng(0).random((5, 4))
        importance = np.array([0.1, 0.2, 0.3, 0.4])
        selected, indices = select_top_features(features, importance, k=99)
        assert selected.shape == (5, 4)
        assert indices.size == 4

    def test_threshold_selection(self) -> None:
        features = np.random.default_rng(0).random((3, 5))
        importance = np.array([0.05, 0.5, 0.9, 0.2, 0.6])
        _, indices = select_top_features(features, importance, threshold=0.5)
        assert set(indices.tolist()) == {1, 2, 4}

    def test_default_selects_all(self) -> None:
        features = np.random.default_rng(0).random((3, 5))
        importance = np.arange(5, dtype=float)
        selected, indices = select_top_features(features, importance)
        assert indices.size == 5
        assert selected.shape == (3, 5)


class TestComputeFeatureImportance:
    def test_variance_importance(self) -> None:
        features = np.array([[0.0, 1.0], [0.0, 5.0], [0.0, 9.0]])
        imp = compute_feature_importance(features, method="variance")
        # First column is constant (zero variance), second varies.
        assert imp[0] == 0.0
        assert imp[1] > 0.0

    def test_correlation_needs_labels_else_falls_back_to_variance(self) -> None:
        features = np.array([[0.0, 1.0], [0.0, 5.0], [0.0, 9.0]])
        # No labels -> falls back to variance (constant first column -> 0).
        imp = compute_feature_importance(features, method="correlation")
        assert imp[0] == 0.0

    def test_correlation_with_labels(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.random((50, 1))
        # Feature 2 is perfectly correlated with labels; feature 1 is noise.
        labels = (x[:, 0] > 0.5).astype(float)
        features = np.column_stack([rng.random(50), labels])
        imp = compute_feature_importance(features, labels, method="correlation")
        assert imp[1] > imp[0]

    def test_mutual_info_runs(self) -> None:
        rng = np.random.default_rng(4)
        features = rng.random((40, 3))
        labels = rng.integers(0, 2, size=40).astype(float)
        imp = compute_feature_importance(features, labels, method="mutual_info")
        assert imp.shape == (3,)
        assert np.all(imp >= 0.0)
