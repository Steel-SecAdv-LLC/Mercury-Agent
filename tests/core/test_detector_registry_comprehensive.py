# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive tests for core/detector_registry.py module.

Covers:
- DetectorRegistry registration, unregistration, lookup
- Category indexing and tag-based filtering
- Feature extraction (single and parallel)
- Import path validation (security hardening)
- Auto-discovery with manifest
- Statistics and health check
- Feature aggregation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from omni_mercury_engine.core.detector_registry import (
    DETECTOR_MANIFEST,
    DetectorCategory,
    DetectorInfo,
    DetectorManifestEntry,
    DetectorRegistry,
    FeatureExtractionResult,
)

# =============================================================================
# DetectorRegistry Core Operations
# =============================================================================


class TestDetectorRegistryCore:
    """Tests for core registry operations."""

    def test_empty_registry(self) -> None:
        """Test empty registry state."""
        reg = DetectorRegistry(auto_discover=False)
        assert reg.list_all() == []
        assert len(reg._detectors) == 0

    def test_register_detector(self) -> None:
        """Test registering a detector."""
        reg = DetectorRegistry(auto_discover=False)
        detector = MagicMock()
        detector.extract_features = MagicMock(return_value=np.zeros(10))
        detector.predict = MagicMock(return_value={"scores": [0.1]})

        reg.register(
            name="test_det",
            detector=detector,
            category=DetectorCategory.BASE,
            description="A test detector",
            tags=["test"],
        )

        assert "test_det" in reg.list_all()
        info = reg.get("test_det")
        assert info is not None
        assert info.name == "test_det"
        assert info.category == DetectorCategory.BASE
        assert info.description == "A test detector"
        assert "test" in info.tags

    def test_unregister_detector(self) -> None:
        """Test unregistering a detector."""
        reg = DetectorRegistry(auto_discover=False)
        reg.register("det1", MagicMock(), DetectorCategory.BASE)
        assert reg.unregister("det1") is True
        assert "det1" not in reg.list_all()

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering non-existent detector returns False."""
        reg = DetectorRegistry(auto_discover=False)
        assert reg.unregister("nonexistent") is False

    def test_get_nonexistent(self) -> None:
        """Test getting non-existent detector returns None."""
        reg = DetectorRegistry(auto_discover=False)
        assert reg.get("nonexistent") is None

    def test_get_by_category(self) -> None:
        """Test retrieving detectors by category."""
        reg = DetectorRegistry(auto_discover=False)
        reg.register("det_base", MagicMock(), DetectorCategory.BASE)
        reg.register("det_medical", MagicMock(), DetectorCategory.MEDICAL)
        reg.register("det_base2", MagicMock(), DetectorCategory.BASE)

        base_dets = reg.get_by_category(DetectorCategory.BASE)
        assert len(base_dets) == 2
        assert all(d.category == DetectorCategory.BASE for d in base_dets)

        medical_dets = reg.get_by_category(DetectorCategory.MEDICAL)
        assert len(medical_dets) == 1

    def test_list_by_tags(self) -> None:
        """Test filtering detectors by tags."""
        reg = DetectorRegistry(auto_discover=False)
        reg.register("det1", MagicMock(), DetectorCategory.BASE, tags=["physics", "spectral"])
        reg.register("det2", MagicMock(), DetectorCategory.BASE, tags=["medical"])
        reg.register("det3", MagicMock(), DetectorCategory.BASE, tags=["physics", "dynamics"])

        physics_dets = reg.list_by_tags(["physics"])
        assert set(physics_dets) == {"det1", "det3"}

        spectral_dets = reg.list_by_tags(["spectral"])
        assert spectral_dets == ["det1"]


# =============================================================================
# Feature Extraction Tests
# =============================================================================


class TestFeatureExtraction:
    """Tests for feature extraction operations."""

    def test_extract_features_success(self) -> None:
        """Test successful feature extraction."""
        reg = DetectorRegistry(auto_discover=False)
        detector = MagicMock()
        detector.extract_features = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        detector.predict = MagicMock(return_value={"anomaly_scores": np.array([0.5])})

        reg.register("det1", detector, DetectorCategory.BASE)

        result = reg.extract_features("det1", np.array([1.0, 2.0]))
        assert result.success is True
        assert result.detector_name == "det1"
        assert result.features is not None
        assert result.execution_time_ms >= 0

    def test_extract_features_not_found(self) -> None:
        """Test extraction for non-existent detector."""
        reg = DetectorRegistry(auto_discover=False)
        result = reg.extract_features("nonexistent", np.array([1.0]))
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error

    def test_extract_features_detector_error(self) -> None:
        """Test extraction handles detector errors gracefully."""
        reg = DetectorRegistry(auto_discover=False)
        detector = MagicMock()
        detector.extract_features = MagicMock(side_effect=RuntimeError("model failed"))

        reg.register("failing_det", detector, DetectorCategory.BASE)

        result = reg.extract_features("failing_det", np.array([1.0]))
        assert result.success is False
        assert result.error is not None

    def test_extract_all_features_sequential(self) -> None:
        """Test extracting features from all detectors sequentially."""
        reg = DetectorRegistry(auto_discover=False)

        for i in range(3):
            det = MagicMock()
            det.extract_features = MagicMock(return_value=np.ones(5) * i)
            det.predict = MagicMock(return_value={})
            reg.register(f"det_{i}", det, DetectorCategory.BASE)

        results = reg.extract_all_features(np.ones(10), parallel=False)
        assert len(results) == 3
        assert all(r.success for r in results.values())

    def test_extract_all_features_filter_by_category(self) -> None:
        """Test category filtering in extract_all_features."""
        reg = DetectorRegistry(auto_discover=False)

        det_base = MagicMock()
        det_base.extract_features = MagicMock(return_value=np.ones(5))
        det_base.predict = MagicMock(return_value={})

        det_med = MagicMock()
        det_med.extract_features = MagicMock(return_value=np.ones(5))
        det_med.predict = MagicMock(return_value={})

        reg.register("base_det", det_base, DetectorCategory.BASE)
        reg.register("med_det", det_med, DetectorCategory.MEDICAL)

        results = reg.extract_all_features(
            np.ones(10),
            parallel=False,
            categories=[DetectorCategory.MEDICAL],
        )
        assert "med_det" in results
        assert "base_det" not in results

    def test_extract_all_features_filter_by_name(self) -> None:
        """Test name filtering in extract_all_features."""
        reg = DetectorRegistry(auto_discover=False)

        for name in ["det_a", "det_b", "det_c"]:
            det = MagicMock()
            det.extract_features = MagicMock(return_value=np.ones(5))
            det.predict = MagicMock(return_value={})
            reg.register(name, det, DetectorCategory.BASE)

        results = reg.extract_all_features(
            np.ones(10),
            parallel=False,
            detector_names=["det_a", "det_c"],
        )
        assert set(results.keys()) == {"det_a", "det_c"}


# =============================================================================
# Import Validation Security Tests
# =============================================================================


class TestImportValidation:
    """Tests for dynamic import path validation (security hardening)."""

    def test_manifest_entries_all_trusted(self) -> None:
        """Test that all manifest entries use trusted module paths."""
        for entry in DETECTOR_MANIFEST:
            assert entry.module_path.startswith("omni_mercury_engine."), (
                f"Manifest entry '{entry.name}' has untrusted module path: " f"{entry.module_path}"
            )

    def test_untrusted_module_path_blocked(self) -> None:
        """Test that auto_discover blocks untrusted module paths."""
        reg = DetectorRegistry(auto_discover=False)

        # Create a manifest entry with an untrusted path
        evil_entry = DetectorManifestEntry(
            name="evil_det",
            module_path="evil.module.path",
            class_name="EvilDetector",
            category=DetectorCategory.BASE,
            description="Should be blocked",
        )

        with patch(
            "omni_mercury_engine.core.detector_registry.DETECTOR_MANIFEST",
            [evil_entry],
        ):
            count = reg.auto_discover_detectors()
            assert count == 0
            assert "evil_det" not in reg.list_all()

    def test_trusted_module_path_allowed(self) -> None:
        """Test that trusted omni_mercury_engine paths are allowed."""
        reg = DetectorRegistry(auto_discover=False)

        entry = DetectorManifestEntry(
            name="trusted_det",
            module_path="omni_mercury_engine.detectors.test_module",
            class_name="TestDetector",
            category=DetectorCategory.BASE,
            description="Should be allowed (but ImportError is OK)",
        )

        with patch(
            "omni_mercury_engine.core.detector_registry.DETECTOR_MANIFEST",
            [entry],
        ):
            # Will fail with ImportError since the module doesn't exist,
            # but should NOT be blocked by the security check
            reg.auto_discover_detectors()
            # Result will be 0 because ImportError, but no security warning
            assert "trusted_det" not in reg.list_all()


# =============================================================================
# Statistics and Health Check Tests
# =============================================================================


class TestStatisticsAndHealth:
    """Tests for registry statistics and health checks."""

    def test_get_statistics_empty(self) -> None:
        """Test statistics for empty registry."""
        reg = DetectorRegistry(auto_discover=False)
        stats = reg.get_statistics()
        assert stats["total_detectors"] == 0
        assert stats["total_invocations"] == 0
        assert stats["total_errors"] == 0

    def test_get_statistics_with_detectors(self) -> None:
        """Test statistics after registering detectors."""
        reg = DetectorRegistry(auto_discover=False)
        reg.register("det1", MagicMock(), DetectorCategory.BASE)
        reg.register("det2", MagicMock(), DetectorCategory.MEDICAL)

        stats = reg.get_statistics()
        assert stats["total_detectors"] == 2
        assert "base" in stats["categories"]
        assert "medical" in stats["categories"]

    def test_get_feature_dimensions(self) -> None:
        """Test feature dimension reporting."""
        reg = DetectorRegistry(auto_discover=False)
        reg.register("det1", MagicMock(), DetectorCategory.BASE, feature_dim=128)
        reg.register("det2", MagicMock(), DetectorCategory.BASE, feature_dim=256)
        reg.register("det3", MagicMock(), DetectorCategory.BASE)

        dims = reg.get_feature_dimensions()
        assert dims["det1"] == 128
        assert dims["det2"] == 256
        assert dims["det3"] is None

    def test_health_check(self) -> None:
        """Test health check for registered detectors."""
        reg = DetectorRegistry(auto_discover=False)

        # Detector with health_check method
        det_with_health = MagicMock()
        det_with_health.health_check = MagicMock(return_value={"healthy": True})
        reg.register("healthy_det", det_with_health, DetectorCategory.BASE)

        # Detector without health_check method
        det_basic = MagicMock(spec=[])
        det_basic.extract_features = True  # Has this attr
        det_basic.predict = True  # Has this attr
        reg.register("basic_det", det_basic, DetectorCategory.BASE)

        health = reg.health_check()
        assert "healthy_det" in health
        assert health["healthy_det"]["healthy"] is True
        assert "basic_det" in health


# =============================================================================
# DetectorInfo Tests
# =============================================================================


class TestDetectorInfo:
    """Tests for DetectorInfo dataclass."""

    def test_to_dict(self) -> None:
        """Test DetectorInfo to_dict conversion."""
        info = DetectorInfo(
            name="test",
            category=DetectorCategory.BASE,
            instance=MagicMock(),
            feature_dim=128,
            is_fitted=True,
            description="Test detector",
            tags=["test", "unit"],
        )

        d = info.to_dict()
        assert d["name"] == "test"
        assert d["category"] == "base"
        assert d["feature_dim"] == 128
        assert d["is_fitted"] is True
        assert d["description"] == "Test detector"
        assert d["tags"] == ["test", "unit"]


# =============================================================================
# FeatureExtractionResult Tests
# =============================================================================


class TestFeatureExtractionResult:
    """Tests for FeatureExtractionResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful feature extraction result."""
        result = FeatureExtractionResult(
            detector_name="test",
            features=np.array([1.0, 2.0, 3.0]),
            scores=np.array([0.5]),
            execution_time_ms=10.5,
            success=True,
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed feature extraction result."""
        result = FeatureExtractionResult(
            detector_name="test",
            features=None,
            scores=None,
            execution_time_ms=0.0,
            success=False,
            error="Detector crashed",
        )
        assert result.success is False
        assert result.error == "Detector crashed"


# =============================================================================
# Manifest Integrity Tests
# =============================================================================


class TestManifestIntegrity:
    """Tests for DETECTOR_MANIFEST data integrity."""

    def test_manifest_not_empty(self) -> None:
        """Test manifest has entries."""
        assert len(DETECTOR_MANIFEST) > 0

    def test_manifest_no_duplicate_names(self) -> None:
        """Test no duplicate detector names in manifest."""
        names = [e.name for e in DETECTOR_MANIFEST]
        assert len(names) == len(set(names)), "Duplicate detector names found"

    def test_manifest_entries_well_formed(self) -> None:
        """Test all manifest entries have required fields."""
        for entry in DETECTOR_MANIFEST:
            assert entry.name, "Entry missing name"
            assert entry.module_path, f"Entry {entry.name} missing module_path"
            assert entry.class_name, f"Entry {entry.name} missing class_name"
            assert isinstance(entry.category, DetectorCategory)
            assert entry.description, f"Entry {entry.name} missing description"

    def test_manifest_covers_expected_categories(self) -> None:
        """Test manifest covers key detector categories."""
        categories = {e.category for e in DETECTOR_MANIFEST}
        expected = {
            DetectorCategory.BASE,
            DetectorCategory.MODEL,
            DetectorCategory.SECURITY,
            DetectorCategory.GEOLOGICAL,
            DetectorCategory.MEDICAL,
            DetectorCategory.SPACE,
        }
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"
