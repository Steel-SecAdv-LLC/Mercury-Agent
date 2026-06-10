# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2 R3: Venn-Abers validity layer — correctness of the wrapper.

The suite measurement (research/governed_fusion/measure_calibration.py) shows
Venn-Abers adds nothing over MCA (marginal Brier ~0, ECE worse) so it is NOT
shipped as the calibration map; these tests pin the wrapper's correctness so the
conclusion rests on a correct implementation.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.conformal_prediction import (
    VennAbersCalibrator,
    _pava_nondecreasing,
)


def test_pava_is_nondecreasing() -> None:
    f = _pava_nondecreasing(np.array([3.0, 1.0, 2.0, 5.0, 4.0]))
    assert np.all(np.diff(f) >= -1e-12)
    # Mass is preserved (isotonic regression averages).
    assert abs(float(np.sum(f)) - 15.0) < 1e-9


def test_venn_abers_interval_is_valid_and_brackets_merged() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 400)
    y = (rng.uniform(0.0, 1.0, 400) < p).astype(int)
    va = VennAbersCalibrator(max_cal=400).fit(p, y)
    assert va._fitted
    p0, p1 = va.predict_interval(p)
    assert np.all(p0 <= p1 + 1e-12)  # a valid multiprobability interval
    merged = va.predict_proba(p)
    assert np.all((merged >= 0.0) & (merged <= 1.0))
    assert np.all((merged >= p0 - 1e-9) & (merged <= p1 + 1e-9))


def test_venn_abers_unfitted_is_identity() -> None:
    p = np.array([0.2, 0.5, 0.8])
    assert np.array_equal(VennAbersCalibrator().predict_proba(p), p)


def test_venn_abers_degenerate_one_class_stays_identity() -> None:
    p = np.linspace(0.1, 0.9, 20)
    y = np.zeros(20, dtype=int)
    va = VennAbersCalibrator().fit(p, y)
    assert not va._fitted
    assert np.array_equal(va.predict_proba(p), p)
