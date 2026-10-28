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

from omni_mercury_engine.core.config import DataCharacteristics
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


def test_temporal_series_caps_the_weight() -> None:
    """Rows that are time steps of one signal break kNN density's iid
    premise: low-d series data must not inherit the tabular gate's high
    weight (measured basis at ``_KNN_WEIGHT_TEMPORAL_CAP``). With no
    continuity measurement the gate fails toward the cap."""
    detector = MercuryAnomalyDetector()
    detector._data_type = DataCharacteristics.TEMPORAL
    assert detector._knn_consec_nn_ratio is None
    assert detector._knn_dimension_weight(3) == detector._KNN_WEIGHT_TEMPORAL_CAP
    # The explicit override still wins over the cap.
    pinned = MercuryAnomalyDetector({"knn_stream_weight": 0.9})
    pinned._data_type = DataCharacteristics.TEMPORAL
    assert pinned._knn_dimension_weight(3) == 0.9


def test_temporal_ordered_instances_keep_the_dimensional_weight() -> None:
    """TEMPORAL-classified data whose rows are independent instances in time
    order (the DTU AIS training files: continuity ratio 5.12-10.86 against
    1.35-2.70 for every series fixture) keeps the dimensional gate weight —
    capping it measured Thesis 0.8934 -> 0.8714 and JMSE 0.8779 -> 0.8640."""
    detector = MercuryAnomalyDetector()
    detector._data_type = DataCharacteristics.TEMPORAL
    detector._knn_consec_nn_ratio = 7.0
    tabular = MercuryAnomalyDetector()
    assert detector._knn_dimension_weight(22) == tabular._knn_dimension_weight(22)
    assert detector._knn_dimension_weight(22) > detector._KNN_WEIGHT_TEMPORAL_CAP

    # End to end: a fitted TEMPORAL detector stores the capped weight.
    rng = np.random.RandomState(7)
    t = np.arange(900)
    cols = []
    for _ in range(3):
        ar = np.zeros(900)
        e = rng.standard_normal(900)
        for i in range(1, 900):
            ar[i] = 0.75 * ar[i - 1] + 0.25 * e[i]
        cols.append(np.sin(2 * np.pi * t / 60.0) + 0.6 * ar)
    series = np.column_stack(cols)
    fitted = MercuryAnomalyDetector().fit(series)
    assert fitted._data_type == DataCharacteristics.TEMPORAL
    assert fitted._knn_fusion_weight is not None
    assert fitted._knn_fusion_weight <= fitted._KNN_WEIGHT_TEMPORAL_CAP


def test_stream_off_leaves_no_weight() -> None:
    detector = MercuryAnomalyDetector({"knn_stream": False})
    detector.fit(np.random.default_rng(1).normal(size=(300, 6)))
    assert detector._knn_fusion_weight is None
