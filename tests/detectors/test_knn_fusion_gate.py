# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The kNN stream's fusion weight follows the dimensional gate contract.

The gate exists because equal z-averaging wasted whichever stream was
stronger: on the ADBench competitive record the kNN stream dominates the
base ensemble at low feature width and the base ensemble owns the
high-dimensional sets (distance concentration). These tests pin the gate's
shape, its caps, the explicit override, and that a fused detector still
honours it end to end.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def test_low_width_trusts_the_knn_stream_more() -> None:
    detector = MercuryAnomalyDetector()
    low = detector._knn_dimension_weight(8)
    mid = detector._knn_dimension_weight(128)
    high = detector._knn_dimension_weight(1555)
    assert low > mid > high
    assert abs(mid - (detector._KNN_WEIGHT_MIN + detector._KNN_WEIGHT_MAX) / 2) < 1e-9


def test_caps_bound_both_regimes() -> None:
    detector = MercuryAnomalyDetector()
    assert detector._knn_dimension_weight(1) <= detector._KNN_WEIGHT_MAX
    assert detector._knn_dimension_weight(100_000) >= detector._KNN_WEIGHT_MIN


def test_explicit_config_weight_wins_and_is_clipped() -> None:
    pinned = MercuryAnomalyDetector({"knn_stream_weight": 0.7})
    assert pinned._knn_dimension_weight(8) == 0.7
    assert pinned._knn_dimension_weight(1555) == 0.7
    clipped = MercuryAnomalyDetector({"knn_stream_weight": 2.0})
    assert clipped._knn_dimension_weight(8) == 1.0


def test_fit_stores_the_weight_and_fusion_moves_with_it() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(400, 8))
    probe = rng.normal(size=(50, 8))
    probe[:5] += 4.0

    fitted = MercuryAnomalyDetector()
    fitted.fit(train)
    assert fitted._knn_fusion_weight is not None
    assert fitted._knn_fusion_weight == fitted._knn_dimension_weight(8)

    # The weight is live in the fusion: pinning it to the extremes changes
    # the combined scores (same data, same fit seedless pipeline).
    base_only = MercuryAnomalyDetector({"knn_stream_weight": 0.0})
    knn_heavy = MercuryAnomalyDetector({"knn_stream_weight": 1.0})
    base_only.fit(train)
    knn_heavy.fit(train)
    s0 = np.asarray(base_only.detect(probe)["scores"], dtype=float)
    s1 = np.asarray(knn_heavy.detect(probe)["scores"], dtype=float)
    assert not np.allclose(s0, s1)


def test_stream_off_leaves_no_weight() -> None:
    detector = MercuryAnomalyDetector({"knn_stream": False})
    detector.fit(np.random.default_rng(1).normal(size=(300, 6)))
    assert detector._knn_fusion_weight is None
