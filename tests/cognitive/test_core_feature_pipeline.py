"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
