# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Novel Class Discovery integration."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.novel_class_discovery import (
    MultiElementBinarization,
    NovelClassDiscovery,
)


class TestMultiElementBinarization:
    """Test MultiElementBinarization class."""

    def test_mebin_initialization(self) -> None:
        """Test MEBin initialization."""
        mebin = MultiElementBinarization()
        assert mebin.rotation_angles == [0, 90, 180, 270]
        assert mebin.binarization_threshold == 0.5

    def test_mebin_custom_config(self) -> None:
        """Test MEBin with custom configuration."""
        config = {"rotation_angles": [0, 45, 90], "binarization_threshold": 0.7}
        mebin = MultiElementBinarization(config)
        assert mebin.rotation_angles == [0, 45, 90]
        assert mebin.binarization_threshold == 0.7

    def test_rotate_to_horizontal_zero_angle(self) -> None:
        """Test rotation with zero angle."""
        mebin = MultiElementBinarization()
        region = np.random.randn(10, 10)
        rotated = mebin.rotate_to_horizontal(region, angle=0)
        np.testing.assert_array_equal(rotated, region)

    def test_rotate_to_horizontal_nonzero_angle(self) -> None:
        """Test rotation with non-zero angle."""
        mebin = MultiElementBinarization()
        region = np.random.randn(10, 10)
        rotated = mebin.rotate_to_horizontal(region, angle=90)
        assert rotated.shape == region.shape

    def test_binarize_above_threshold(self) -> None:
        """Test binarization for values above threshold."""
        mebin = MultiElementBinarization({"binarization_threshold": 0.5})
        mask = np.array([0.3, 0.7, 0.9, 0.1])
        binary = mebin.binarize(mask)
        expected = np.array([0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(binary, expected)

    def test_binarize_all_above(self) -> None:
        """Test binarization when all values above threshold."""
        mebin = MultiElementBinarization({"binarization_threshold": 0.5})
        mask = np.array([0.6, 0.8, 1.0])
        binary = mebin.binarize(mask)
        np.testing.assert_array_equal(binary, np.ones(3))

    def test_binarize_all_below(self) -> None:
        """Test binarization when all values below threshold."""
        mebin = MultiElementBinarization({"binarization_threshold": 0.5})
        mask = np.array([0.1, 0.2, 0.4])
        binary = mebin.binarize(mask)
        np.testing.assert_array_equal(binary, np.zeros(3))

    def test_process_multi_element(self) -> None:
        """Test processing multiple elements."""
        mebin = MultiElementBinarization()
        regions = [np.random.randn(5, 5), np.random.randn(5, 5), np.random.randn(5, 5)]
        processed = mebin.process_multi_element(regions)
        assert len(processed) == 3
        assert all(isinstance(p, np.ndarray) for p in processed)

    def test_process_empty_list(self) -> None:
        """Test processing empty list of regions."""
        mebin = MultiElementBinarization()
        processed = mebin.process_multi_element([])
        assert len(processed) == 0


class TestNovelClassDiscovery:
    """Test NovelClassDiscovery class."""

    def test_ncd_initialization(self) -> None:
        """Test NCD initialization."""
        ncd = NovelClassDiscovery()
        assert ncd.enable_mebin is True
        assert ncd.low_semantics_mode is True
        assert ncd.non_prominence_mode is True
        assert ncd.num_clusters == 5

    def test_ncd_custom_config(self) -> None:
        """Test NCD with custom configuration."""
        config = {
            "enable_mebin": False,
            "low_semantics_mode": False,
            "non_prominence_mode": False,
            "num_clusters": 3,
        }
        ncd = NovelClassDiscovery(config)
        assert ncd.enable_mebin is False
        assert ncd.low_semantics_mode is False
        assert ncd.non_prominence_mode is False
        assert ncd.num_clusters == 3

    def test_mebin_initialization_when_enabled(self) -> None:
        """Test MEBin is initialized when enabled."""
        ncd = NovelClassDiscovery({"enable_mebin": True})
        assert ncd.mebin is not None
        assert isinstance(ncd.mebin, MultiElementBinarization)

    def test_mebin_not_initialized_when_disabled(self) -> None:
        """Test MEBin is not initialized when disabled."""
        ncd = NovelClassDiscovery({"enable_mebin": False})
        assert ncd.mebin is None

    def test_extract_anomaly_features_shape(self) -> None:
        """Test anomaly feature extraction shape."""
        ncd = NovelClassDiscovery()
        images = np.random.randn(10, 32, 32, 3)
        masks = np.random.rand(10, 32, 32)

        features = ncd.extract_anomaly_features(images, masks)
        assert features.shape == (10, 5)

    def test_extract_anomaly_features_grayscale(self) -> None:
        """Test feature extraction with grayscale images."""
        ncd = NovelClassDiscovery()
        images = np.random.randn(5, 32, 32)
        masks = np.random.rand(5, 32, 32)

        features = ncd.extract_anomaly_features(images, masks)
        assert features.shape == (5, 5)

    def test_discover_novel_classes_basic(self) -> None:
        """Test basic novel class discovery."""
        ncd = NovelClassDiscovery({"num_clusters": 3})
        images = np.random.randn(20, 32, 32, 3)
        masks = np.random.rand(20, 32, 32)

        results = ncd.discover_novel_classes(images, masks)

        assert "class_assignments" in results
        assert "discovered_classes" in results
        assert "num_classes" in results
        assert "cluster_centers" in results
        assert results["num_classes"] == 3

    def test_discover_classes_assignments_shape(self) -> None:
        """Test class assignments shape."""
        ncd = NovelClassDiscovery({"num_clusters": 4})
        images = np.random.randn(15, 32, 32, 3)
        masks = np.random.rand(15, 32, 32)

        results = ncd.discover_novel_classes(images, masks)
        assert results["class_assignments"].shape == (15,)

    def test_discover_classes_labels_format(self) -> None:
        """Test discovered class labels format."""
        ncd = NovelClassDiscovery({"num_clusters": 3})
        images = np.random.randn(10, 32, 32, 3)
        masks = np.random.rand(10, 32, 32)

        results = ncd.discover_novel_classes(images, masks)
        assert len(results["discovered_classes"]) == 3
        assert all("novel_class_" in label for label in results["discovered_classes"])

    def test_discover_classes_method_label(self) -> None:
        """Test method label in results."""
        ncd = NovelClassDiscovery()
        images = np.random.randn(10, 32, 32, 3)
        masks = np.random.rand(10, 32, 32)

        results = ncd.discover_novel_classes(images, masks)
        assert results["method"] == "AnomalyNCD"

    def test_classify_new_anomaly_without_discovery_raises_error(self) -> None:
        """Test classification without discovery raises error."""
        ncd = NovelClassDiscovery()
        image = np.random.randn(32, 32, 3)
        mask = np.random.rand(32, 32)

        with pytest.raises(ValueError, match="Must discover classes first"):
            ncd.classify_new_anomaly(image, mask)

    def test_classify_new_anomaly_basic(self) -> None:
        """Test classifying new anomaly."""
        ncd = NovelClassDiscovery({"num_clusters": 3})

        images = np.random.randn(20, 32, 32, 3)
        masks = np.random.rand(20, 32, 32)
        ncd.discover_novel_classes(images, masks)

        new_image = np.random.randn(32, 32, 3)
        new_mask = np.random.rand(32, 32)
        results = ncd.classify_new_anomaly(new_image, new_mask)

        assert "predicted_class" in results
        assert "predicted_class_idx" in results
        assert "confidence" in results

    def test_classify_confidence_range(self) -> None:
        """Test classification confidence is in valid range."""
        ncd = NovelClassDiscovery({"num_clusters": 3})

        images = np.random.randn(20, 32, 32, 3)
        masks = np.random.rand(20, 32, 32)
        ncd.discover_novel_classes(images, masks)

        new_image = np.random.randn(32, 32, 3)
        new_mask = np.random.rand(32, 32)
        results = ncd.classify_new_anomaly(new_image, new_mask)

        assert 0.0 <= results["confidence"] <= 1.0

    def test_classify_predicted_class_idx_valid(self) -> None:
        """Test predicted class index is valid."""
        ncd = NovelClassDiscovery({"num_clusters": 4})

        images = np.random.randn(20, 32, 32, 3)
        masks = np.random.rand(20, 32, 32)
        ncd.discover_novel_classes(images, masks)

        new_image = np.random.randn(32, 32, 3)
        new_mask = np.random.rand(32, 32)
        results = ncd.classify_new_anomaly(new_image, new_mask)

        assert 0 <= results["predicted_class_idx"] < 4

    def test_get_class_statistics_basic(self) -> None:
        """Test getting class statistics."""
        ncd = NovelClassDiscovery()
        class_assignments = np.array([0, 1, 0, 2, 1, 0, 2, 1, 0])

        stats = ncd.get_class_statistics(class_assignments)

        assert "num_samples_per_class" in stats
        assert "total_samples" in stats
        assert "class_distribution" in stats
        assert stats["total_samples"] == 9

    def test_get_class_statistics_distribution(self) -> None:
        """Test class distribution sums to 1."""
        ncd = NovelClassDiscovery()
        class_assignments = np.array([0, 1, 2, 0, 1, 2])

        stats = ncd.get_class_statistics(class_assignments)
        distribution = np.array(stats["class_distribution"])

        np.testing.assert_almost_equal(np.sum(distribution), 1.0)

    def test_get_class_statistics_most_common(self) -> None:
        """Test most common class identification."""
        ncd = NovelClassDiscovery()
        class_assignments = np.array([0, 0, 0, 1, 1, 2])

        stats = ncd.get_class_statistics(class_assignments)
        assert stats["most_common_class"] == 0

    def test_get_class_statistics_least_common(self) -> None:
        """Test least common class identification."""
        ncd = NovelClassDiscovery()
        class_assignments = np.array([0, 0, 0, 1, 1, 2])

        stats = ncd.get_class_statistics(class_assignments)
        assert stats["least_common_class"] == 2

    def test_mebin_enabled_flag_in_results(self) -> None:
        """Test MEBin enabled flag in discovery results."""
        ncd = NovelClassDiscovery({"enable_mebin": True})
        images = np.random.randn(10, 32, 32, 3)
        masks = np.random.rand(10, 32, 32)

        results = ncd.discover_novel_classes(images, masks)
        assert results["mebin_enabled"] is True

    def test_low_semantics_mode_config(self) -> None:
        """Test low semantics mode configuration."""
        ncd = NovelClassDiscovery({"low_semantics_mode": True})
        assert ncd.low_semantics_mode is True

    def test_non_prominence_mode_config(self) -> None:
        """Test non-prominence mode configuration."""
        ncd = NovelClassDiscovery({"non_prominence_mode": True})
        assert ncd.non_prominence_mode is True
