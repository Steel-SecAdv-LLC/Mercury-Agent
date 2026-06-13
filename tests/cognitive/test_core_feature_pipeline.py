# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test core feature pipeline."""

from __future__ import annotations

from omni_mercury_engine.core.feature_pipeline import (
    FeaturePipeline,
    FeatureStandardizer,
    FeatureStore,
)


def test_feature_pipeline_instantiation() -> None:
    fp = FeaturePipeline()
    assert fp is not None
    assert hasattr(fp, "fit")
    assert hasattr(fp, "transform")


def test_feature_standardizer_instantiation() -> None:
    fs = FeatureStandardizer()
    assert fs is not None


def test_feature_store_instantiation() -> None:
    store = FeatureStore()
    assert store is not None


def test_importable_from_core() -> None:
    from omni_mercury_engine.core import (
        FeaturePipeline as FP,
        FeatureStandardizer as FS,
        FeatureStore as FSt,
    )

    assert FP is FeaturePipeline
    assert FS is FeatureStandardizer
    assert FSt is FeatureStore
