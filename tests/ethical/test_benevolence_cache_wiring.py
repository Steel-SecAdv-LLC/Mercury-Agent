# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The engine ethics boundary uses the LRU benevolence cache on the live path.

``CachedBenevolenceScorer`` was a complete-but-orphaned module (tested in
isolation, never on the runtime path). These pin that the engine now wraps its
boundary scorer with it by default and that repeated same-shape/domain
``detect_with_fusion`` calls actually hit the cache -- i.e. the cache is
exercised at runtime, not merely importable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.cognitive.benevolence_cache import CachedBenevolenceScorer
from omni_mercury_engine.cognitive.ethical_bounding import BenevolenceScorer


def _engine(**kwargs):
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(
        mode="fusion", device="cpu", require_explicit_fit=False, **kwargs
    )


def test_boundary_scorer_is_cached_by_default() -> None:
    engine = _engine()
    assert isinstance(engine._boundary_scorer, CachedBenevolenceScorer)
    # The cache wraps a real scorer, not a second cache.
    assert isinstance(engine._boundary_scorer.underlying_scorer, BenevolenceScorer)


def test_repeat_detection_hits_the_boundary_cache() -> None:
    engine = _engine()
    rng = np.random.default_rng(0)

    # Two detections with the same data shape and domain produce an identical
    # (action, context) at the ethics boundary -> first is a miss, second a hit.
    engine.detect_with_fusion(rng.normal(0, 1, (16, 6)), domain="general")
    engine.detect_with_fusion(rng.normal(0, 1, (16, 6)), domain="general")

    stats = engine._boundary_scorer.stats
    assert stats["misses"] >= 1
    assert stats["hits"] >= 1


def test_cache_disabled_uses_plain_scorer() -> None:
    engine = _engine(cache_ethical_decisions=False)
    assert isinstance(engine._boundary_scorer, BenevolenceScorer)
    assert not isinstance(engine._boundary_scorer, CachedBenevolenceScorer)
