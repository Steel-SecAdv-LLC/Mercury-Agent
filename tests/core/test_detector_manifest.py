# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the data-driven detector manifest and auto-discovery refactor.

Verifies that:
- DETECTOR_MANIFEST is a well-formed list with no duplicates
- auto_discover_detectors uses the manifest correctly
- DetectorManifestEntry dataclass validates fields
- The manifest covers all expected detector categories
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omni_mercury_engine.core.detector_registry import (
    DETECTOR_MANIFEST,
    DetectorCategory,
    DetectorManifestEntry,
    DetectorRegistry,
)


class TestDetectorManifestEntry:
    """Test the DetectorManifestEntry dataclass."""

    def test_basic_creation(self) -> None:
        entry = DetectorManifestEntry(
            name="test_det",
            module_path="some.module",
            class_name="TestDetector",
            category=DetectorCategory.BASE,
            description="A test detector",
        )
        assert entry.name == "test_det"
        assert entry.module_path == "some.module"
        assert entry.class_name == "TestDetector"
        assert entry.category == DetectorCategory.BASE
        assert entry.description == "A test detector"
        assert entry.feature_dim is None
        assert entry.tags == []

    def test_creation_with_optional_fields(self) -> None:
        entry = DetectorManifestEntry(
            name="tagged",
            module_path="a.b",
            class_name="X",
            category=DetectorCategory.GEOLOGICAL,
            description="desc",
            feature_dim=20,
            tags=["disaster", "weather"],
        )
        assert entry.feature_dim == 20
        assert entry.tags == ["disaster", "weather"]

    def test_frozen(self) -> None:
        entry = DetectorManifestEntry(
            name="x",
            module_path="a",
            class_name="B",
            category=DetectorCategory.BASE,
            description="d",
        )
        with pytest.raises(AttributeError):
            entry.name = "y"  # type: ignore[misc]


class TestDetectorManifest:
    """Test the global DETECTOR_MANIFEST list."""

    def test_manifest_is_nonempty(self) -> None:
        assert len(DETECTOR_MANIFEST) > 0

    def test_manifest_names_unique(self) -> None:
        names = [e.name for e in DETECTOR_MANIFEST]
        assert len(names) == len(
            set(names)
        ), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_all_entries_are_manifest_entry(self) -> None:
        for entry in DETECTOR_MANIFEST:
            assert isinstance(entry, DetectorManifestEntry)

    def test_all_entries_have_required_fields(self) -> None:
        for entry in DETECTOR_MANIFEST:
            assert entry.name, "Empty name in manifest"
            assert entry.module_path, f"Empty module_path for {entry.name}"
            assert entry.class_name, f"Empty class_name for {entry.name}"
            assert isinstance(
                entry.category, DetectorCategory
            ), f"Invalid category for {entry.name}"
            assert entry.description, f"Empty description for {entry.name}"

    def test_categories_covered(self) -> None:
        """Verify that the manifest covers the expected broad detector categories."""
        categories_present = {e.category for e in DETECTOR_MANIFEST}
        expected = {
            DetectorCategory.BASE,
            DetectorCategory.MODEL,
            DetectorCategory.SECURITY,
            DetectorCategory.GEOLOGICAL,
            DetectorCategory.SPACE,
            DetectorCategory.MEDICAL,
            DetectorCategory.VISUAL,
            DetectorCategory.VLM,
            DetectorCategory.FOUNDATION,
            DetectorCategory.PHYSICS,
        }
        missing = expected - categories_present
        assert not missing, f"Missing categories in manifest: {missing}"

    def test_manifest_count_matches_original(self) -> None:
        """The manifest should have at least 40 entries (original had ~45 detectors)."""
        assert len(DETECTOR_MANIFEST) >= 40


class TestAutoDiscoverDetectorsRefactored:
    """Test that auto_discover_detectors works with the manifest."""

    def test_returns_zero_when_all_imports_fail(self) -> None:
        registry = DetectorRegistry()
        with patch("builtins.__import__", side_effect=ImportError("mocked")):
            count = registry.auto_discover_detectors()
        assert count == 0
        assert len(registry.list_all()) == 0

    def test_registers_detector_on_successful_import(self) -> None:
        registry = DetectorRegistry()

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()

        mock_module = MagicMock()
        mock_module.MercuryAnomalyDetector = mock_cls

        # Patch just the first manifest entry's module, fail the rest
        def selective_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "omni_mercury_engine.detectors.statistical":
                return mock_module
            raise ImportError(f"mocked: {name}")

        with patch("builtins.__import__", side_effect=selective_import):
            count = registry.auto_discover_detectors()

        assert count == 1
        assert "statistical" in registry.list_all()

    def test_auto_discover_idempotent_registration(self) -> None:
        """Calling auto_discover twice should re-register (overwrite), not crash."""
        registry = DetectorRegistry()

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        mock_module = MagicMock()
        mock_module.MercuryAnomalyDetector = mock_cls

        def selective_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "omni_mercury_engine.detectors.statistical":
                return mock_module
            raise ImportError(f"mocked: {name}")

        with patch("builtins.__import__", side_effect=selective_import):
            count1 = registry.auto_discover_detectors()
            count2 = registry.auto_discover_detectors()

        assert count1 == count2 == 1

    def test_non_import_error_logged_as_warning(self) -> None:
        """Non-ImportError exceptions should be caught and logged as warnings."""
        registry = DetectorRegistry()

        def always_fail(name: str, *args: object, **kwargs: object) -> object:
            if name == "omni_mercury_engine.detectors.statistical":
                raise RuntimeError("class init failed")
            raise ImportError("mocked")

        with patch("builtins.__import__", side_effect=always_fail):
            # Should not raise
            count = registry.auto_discover_detectors()

        assert count == 0

    def test_feature_dim_and_tags_propagated(self) -> None:
        """Verify that feature_dim and tags from manifest entries are registered."""
        registry = DetectorRegistry()

        # Find the tornado entry (it has feature_dim=20 and tags)
        tornado_entry = next(e for e in DETECTOR_MANIFEST if e.name == "tornado")
        assert tornado_entry.feature_dim == 20
        assert "disaster" in tornado_entry.tags

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        mock_module = MagicMock()
        setattr(mock_module, tornado_entry.class_name, mock_cls)

        def selective_import(name: str, *args: object, **kwargs: object) -> object:
            if name == tornado_entry.module_path:
                return mock_module
            raise ImportError(f"mocked: {name}")

        with patch("builtins.__import__", side_effect=selective_import):
            registry.auto_discover_detectors()

        info = registry.get("tornado")
        assert info is not None
        assert info.feature_dim == 20
        assert "disaster" in info.tags
