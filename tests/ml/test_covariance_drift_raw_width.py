# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The covariance drift test checks a batch against the reference's *raw* width.

The detector stores its reference after dropping constant columns and, past
``MAX_DIMENSION`` features, projecting onto the leading principal axes. The
width check compared that projected reference with the raw batch, so any
reference with a constant column or more than 64 features raised
``DriftIncomparableError`` on every batch, which the engine reports as
``incomparable``: drift detection was permanently off for such data.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.ml.drift import (
    CovarianceDriftDetector,
    DriftIncomparableError,
    EnsembleDriftDetector,
)


def _with_constant_column(n: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n, 6))
    x[:, 2] = 3.0  # e.g. a sensor stuck at a fixed reading in training
    return x


def test_a_constant_reference_column_does_not_make_batches_incomparable() -> None:
    detector = CovarianceDriftDetector(n_null=50).fit(_with_constant_column(600, seed=0))
    assert detector.unavailable_reason is None
    found = detector.detect(_with_constant_column(200, seed=1))
    assert "skipped" not in found.details
    assert found.details["dimension"] == 5
    assert found.is_drift is False


def test_more_than_max_dimension_features_are_comparable() -> None:
    d = CovarianceDriftDetector.MAX_DIMENSION + 16  # 80 features
    rng = np.random.default_rng(2)
    detector = CovarianceDriftDetector(n_null=30).fit(rng.normal(size=(600, d)))
    found = detector.detect(rng.normal(size=(200, d)))
    assert "skipped" not in found.details
    assert found.details["dimension"] == CovarianceDriftDetector.MAX_DIMENSION


def test_a_raw_width_mismatch_still_raises() -> None:
    detector = CovarianceDriftDetector(n_null=30).fit(_with_constant_column(600, seed=3))
    with pytest.raises(DriftIncomparableError, match="the reference has 6"):
        detector.detect(np.random.default_rng(4).normal(size=(200, 5)))  # projected width


def test_the_ensemble_runs_on_a_reference_with_a_constant_column() -> None:
    detector = EnsembleDriftDetector().fit(_with_constant_column(600, seed=5))
    found = detector.detect(_with_constant_column(200, seed=6))
    assert found.details["covariance_result"] is not None
    assert "skipped" not in found.details["covariance_result"]["details"]
