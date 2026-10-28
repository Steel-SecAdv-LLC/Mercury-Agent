# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean-shift drift along principal axes, which per-feature tests dilute.

In correlated data a shift along a low-variance direction moves each column by
a fraction of its spread. ``benchmarks/drift_power.py`` measures it: with 8
features correlated at 0.9, a one-SD shift along the weakest axis was flagged
in 0-8% of windows by the KS, PSI and covariance ensemble and in 100% once the
principal-axis test was added.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pytest

from omni_mercury_engine.ml.drift import (
    DriftIncomparableError,
    DriftSeverity,
    DriftType,
    EnsembleDriftDetector,
    PrincipalAxisDriftDetector,
    create_drift_detector,
)

Array: TypeAlias = np.ndarray[Any, Any]


def _scenario(d: int, rho: float, seed: int = 0) -> tuple[Array, Array, Array]:
    """Return a correlated sampler matrix, and the weakest axis with its std."""
    cov = np.asarray(rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d))))
    values, vectors = np.linalg.eigh(cov)
    return np.linalg.cholesky(cov), vectors[:, 0], np.sqrt(values[0])


def _draw(rng: np.random.Generator, chol: Array, rows: int, shift: Array | float = 0.0) -> Array:
    return np.asarray(rng.normal(size=(rows, chol.shape[0])) @ chol.T + shift)


@pytest.fixture(scope="module")
def correlated() -> tuple[Array, Array, Array, float]:
    chol, weakest, weakest_std = _scenario(8, 0.9)
    reference = _draw(np.random.default_rng(0), chol, 2000)
    return reference, chol, weakest, float(weakest_std)


def test_flags_a_shift_along_the_weakest_axis_that_the_other_tests_miss(
    correlated: tuple[Array, Array, Array, float],
) -> None:
    reference, chol, weakest, weakest_std = correlated
    window = _draw(np.random.default_rng(1), chol, 1000, 1.0 * weakest_std * weakest)

    # No single column moves by more than about a tenth of its spread.
    assert np.abs(1.0 * weakest_std * weakest).max() < 0.15

    without = EnsembleDriftDetector(principal_axes=False).fit(reference)
    assert not without.detect(window).is_drift

    full = EnsembleDriftDetector().fit(reference)
    result = full.detect(window)
    assert result.is_drift
    assert result.drift_type is DriftType.PRINCIPAL_AXIS_DRIFT
    assert result.severity is DriftSeverity.CRITICAL
    assert result.details["principal_axis_drift"] is True
    assert "principal axis" in result.message
    axis = result.details["principal_axis_result"]["details"]
    assert axis["worst_axis"] == axis["n_axes"] - 1  # the weakest axis is the last


def test_false_alarm_rate_stays_near_its_level(
    correlated: tuple[Array, Array, Array, float],
) -> None:
    """0.01 family-wise: 100 clean windows, a handful at most (expected one)."""
    reference, chol, _, _ = correlated
    detector = PrincipalAxisDriftDetector().fit(reference)
    rng = np.random.default_rng(2)
    flagged = sum(detector.detect(_draw(rng, chol, 200)).is_drift for _ in range(100))
    assert flagged <= 5


def test_axis_aligned_shift_in_uncorrelated_data_is_still_caught() -> None:
    chol, _, _ = _scenario(6, 0.0)
    rng = np.random.default_rng(3)
    detector = PrincipalAxisDriftDetector().fit(_draw(rng, chol, 1500))
    shift = np.zeros(6)
    shift[2] = 0.8
    assert detector.detect(_draw(rng, chol, 200, shift)).is_drift


def test_axes_and_reference_distribution_come_from_different_rows() -> None:
    """128 correlated features from 400 reference rows: axes fitted on the
    same rows they are tested against flagged every window (100% false alarms);
    the split keeps them quiet."""
    chol, _, _ = _scenario(128, 0.95)
    rng = np.random.default_rng(4)
    detector = PrincipalAxisDriftDetector().fit(_draw(rng, chol, 400))
    flagged = [detector.detect(_draw(rng, chol, 300)).is_drift for _ in range(8)]
    assert not any(flagged)


def test_not_fitted_and_too_small_inputs_are_skipped_not_guessed() -> None:
    detector = PrincipalAxisDriftDetector()
    result = detector.detect(np.zeros((50, 3)))
    assert not result.is_drift and result.details["skipped"] == "not fitted"

    assert "at least two features" in str(
        PrincipalAxisDriftDetector().fit(np.arange(100.0)).unavailable_reason
    )
    few = PrincipalAxisDriftDetector().fit(np.random.default_rng(5).normal(size=(30, 4)))
    assert few.unavailable_reason is not None and "needs at least" in few.unavailable_reason
    assert few.detect(np.zeros((40, 4))).details["skipped"] == few.unavailable_reason

    fitted = PrincipalAxisDriftDetector().fit(np.random.default_rng(6).normal(size=(200, 4)))
    assert "window has 5 rows" in fitted.detect(np.zeros((5, 4))).details["skipped"]


def test_constant_reference_has_no_axes_to_test() -> None:
    detector = PrincipalAxisDriftDetector().fit(np.ones((200, 3)))
    assert detector.unavailable_reason == "reference has no variance"
    assert not detector.detect(np.ones((50, 3))).is_drift


def test_constant_column_does_not_break_the_axes() -> None:
    rng = np.random.default_rng(7)
    reference = np.column_stack([rng.normal(size=(300, 3)), np.full(300, 5.0)])
    detector = PrincipalAxisDriftDetector().fit(reference)
    window = np.column_stack([rng.normal(size=(100, 3)), np.full(100, 5.0)])
    result = detector.detect(window)
    assert result.details["n_axes"] == 3 and not result.is_drift


def test_non_finite_rows_are_dropped_and_width_must_match(
    correlated: tuple[Array, Array, Array, float],
) -> None:
    reference, chol, _, _ = correlated
    detector = PrincipalAxisDriftDetector().fit(reference)
    window = _draw(np.random.default_rng(8), chol, 120)
    window[:20, 3] = np.nan
    assert detector.detect(window).details["window_size"] == 100
    with pytest.raises(DriftIncomparableError):
        detector.detect(window[:, :5])


def test_reports_the_ladder_the_other_p_value_tests_use() -> None:
    from omni_mercury_engine.detectors.drift import _severity_for_p_value

    ladder = [_severity_for_p_value(p, 0.05) for p in (0.2, 0.04, 0.004, 0.0004, 0.00001)]
    assert ladder == [
        DriftSeverity.NONE,
        DriftSeverity.LOW,
        DriftSeverity.MEDIUM,
        DriftSeverity.HIGH,
        DriftSeverity.CRITICAL,
    ]


def test_ensemble_without_the_axis_test_keeps_its_previous_behaviour(
    correlated: tuple[Array, Array, Array, float],
) -> None:
    reference, chol, _, _ = correlated
    ensemble = EnsembleDriftDetector(principal_axes=False).fit(reference)
    assert ensemble.axis_detector is None
    result = ensemble.detect(_draw(np.random.default_rng(9), chol, 200))
    assert result.details["principal_axis_result"] is None
    assert result.details["principal_axis_drift"] is False
    assert result.message == "No ensemble drift detected"


def test_ensemble_survives_an_axis_test_that_cannot_run() -> None:
    """A single feature has no principal axes; the other tests still decide."""
    rng = np.random.default_rng(10)
    ensemble = EnsembleDriftDetector().fit(rng.normal(size=(500, 1)))
    result = ensemble.detect(rng.normal(size=(100, 1)) + 3.0)
    assert result.is_drift and result.drift_type is DriftType.DATA_DRIFT
    assert "skipped" in result.details["principal_axis_result"]["details"]


def test_factory_builds_the_detector() -> None:
    assert isinstance(create_drift_detector("principal_axes"), PrincipalAxisDriftDetector)


@pytest.mark.parametrize(("n_reference", "window", "features"), [(600, 40, 5), (3000, 500, 8)])
@pytest.mark.parametrize("decimals", [None, 1])
def test_statistics_and_p_value_match_scipy_per_axis(
    n_reference: int, window: int, features: int, decimals: int | None
) -> None:
    """The pre-sorted statistic equals ``ks_2samp``'s on every axis (ties included),
    and the worst axis carries the smallest p-value that scipy finds."""
    from scipy import stats

    rng = np.random.default_rng(n_reference)
    mix = rng.normal(size=(features, features))
    detector = PrincipalAxisDriftDetector().fit(rng.normal(size=(n_reference, features)) @ mix)
    current = rng.normal(size=(window, features)) @ mix + 0.4
    assert detector._center is not None and detector._axes is not None
    assert detector._reference_scores is not None
    scores = (current - detector._center) @ detector._axes
    if decimals is not None:  # heavy ties in both samples
        scores = np.round(scores, decimals)
        detector._reference_scores = np.sort(np.round(detector._reference_scores, decimals), axis=0)

    got = detector._ks_statistics(scores)
    tests = [
        stats.ks_2samp(detector._reference_scores[:, j], scores[:, j])
        for j in range(scores.shape[1])
    ]
    np.testing.assert_allclose(got, [t.statistic for t in tests], rtol=0, atol=1e-12)
    worst = int(np.argmax(got))
    assert tests[worst].pvalue == pytest.approx(min(t.pvalue for t in tests), rel=1e-12)
