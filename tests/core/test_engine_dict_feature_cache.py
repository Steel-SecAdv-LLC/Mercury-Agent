# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression test: distinct dict inputs must not collide in the feature cache.

The fusion feature extractors previously keyed dict inputs by a constant
placeholder (``np.array([0])``) because the feature cache keys on array bytes.
Every distinct dict payload therefore collapsed onto a single cache entry, so
the second and later dicts received the first payload's stale features. Dict
inputs now bypass the cache and are computed fresh (array/tensor inputs still
cache), which this test pins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.engine import OmniMercuryEngine


class _DictFeatureDetector(BaseDetector):
    """Fitted detector whose feature/score depend on ``data['value']``."""

    def __init__(self) -> None:
        super().__init__({"threshold": 0.5})
        self._is_fitted = True

    def fit(self, data: Any) -> _DictFeatureDetector:
        self._is_fitted = True
        return self

    def _value(self, data: Any) -> float:
        if isinstance(data, dict):
            return float(data["value"])
        return float(np.asarray(data).sum())

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        return np.array([[self._value(data)]], dtype=np.float32)

    def detect(self, data: Any) -> dict[str, Any]:
        return {"scores": [0.0], "is_anomaly": False}


def test_distinct_dict_payloads_get_distinct_features() -> None:
    engine = OmniMercuryEngine(
        mode="fusion",
        auto_load_checkpoint=False,
        require_explicit_fit=True,
    )
    engine.register_detector("dictdet", _DictFeatureDetector())

    features_a, _, _ = engine._extract_detector_features({"value": 1.0})
    features_b, _, _ = engine._extract_detector_features({"value": 2.0})

    # Under the constant-key cache collision, features_b would have echoed the
    # first payload's value (1.0) instead of 2.0.
    assert float(features_a["dictdet"][0, 0]) == 1.0
    assert float(features_b["dictdet"][0, 0]) == 2.0
