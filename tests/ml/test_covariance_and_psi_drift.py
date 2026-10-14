# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Covariance-structure drift and the corrected PSI drift test."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.riemannian_optimization import SPDManifold
from omni_mercury_engine.ml.drift import (
    CovarianceDriftDetector,
    DriftType,
    EnsembleDriftDetector,
    PopulationStabilityIndexDetector,
    create_drift_detector,
)

D = 6


def _corr(rho: float, d: int = D) -> np.ndarray:
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    return np.random.default_rng(0).multivariate_normal(np.zeros(D), _corr(0.6), 1500)


@pytest.fixture(scope="module")
def cov_detector(reference: np.ndarray) -> CovarianceDriftDetector:
    return CovarianceDriftDetector(n_null=100).fit(reference)


def test_spd_distance_matches_matrix_function_definition() -> None:
    rng = np.random.default_rng(1)
    a = np.cov(rng.standard_normal((200, 5)), rowvar=False)
    b = np.cov(rng.standard_normal((200, 5)) @ np.diag([1, 2, 1, 0.5, 1]), rowvar=False)
    w, v = np.linalg.eigh(a)
    a_inv_sqrt = (v / np.sqrt(w)) @ v.T
    expected = np.linalg.norm(np.log(np.linalg.eigvalsh(a_inv_sqrt @ b @ a_inv_sqrt)))
    assert SPDManifold(5).geodesic_distance(a, b) == pytest.approx(expected, rel=1e-10)


def test_marginal_preserving_decorrelation_is_detected(
    cov_detector: CovarianceDriftDetector,
) -> None:
    rng = np.random.default_rng(2)
    window = rng.standard_normal((300, D))  # every marginal still N(0, 1)
    result = cov_detector.detect(window)
    assert result.is_drift
    assert result.drift_type == DriftType.COVARIANCE_DRIFT
    assert result.p_value < 0.05


def test_no_drift_on_reference_distribution(cov_detector: CovarianceDriftDetector) -> None:
    rng = np.random.default_rng(3)
    flags = [
        cov_detector.detect(rng.multivariate_normal(np.zeros(D), _corr(0.6), 300)).is_drift
        for _ in range(20)
    ]
    assert sum(flags) <= 3


def test_small_window_is_skipped_not_guessed(cov_detector: CovarianceDriftDetector) -> None:
    result = cov_detector.detect(np.random.default_rng(4).standard_normal((5, D)))
    assert not result.is_drift
    assert "skipped" in result.details


def test_single_feature_reference_is_unavailable() -> None:
    det = CovarianceDriftDetector().fit(np.random.default_rng(5).standard_normal((500, 1)))
    assert det.unavailable_reason is not None
    assert not det.detect(np.random.default_rng(6).standard_normal((100, 1))).is_drift


def test_null_is_deterministic_and_cached(reference: np.ndarray) -> None:
    window = np.random.default_rng(7).standard_normal((250, D))
    a = CovarianceDriftDetector(n_null=50).fit(reference)
    b = CovarianceDriftDetector(n_null=50).fit(reference)
    assert a.detect(window).p_value == b.detect(window).p_value
    assert len(a._null_cache) == 1


def test_wide_data_is_projected() -> None:
    rng = np.random.default_rng(8)
    ref = rng.standard_normal((400, CovarianceDriftDetector.MAX_DIMENSION + 16))
    det = CovarianceDriftDetector(n_null=20).fit(ref)
    assert det._projection is not None
    assert det._projection.shape[1] == CovarianceDriftDetector.MAX_DIMENSION
    assert det.unavailable_reason is None


def test_ensemble_reports_covariance_only_drift(reference: np.ndarray) -> None:
    ens = EnsembleDriftDetector().fit(reference)
    ens.cov_detector = CovarianceDriftDetector(n_null=100).fit(reference)
    result = ens.detect(np.random.default_rng(9).standard_normal((300, D)))
    assert result.is_drift
    assert result.drift_type == DriftType.COVARIANCE_DRIFT
    assert result.details["covariance_drift"] is True
    assert result.details["marginal_drift"] is False


def test_marginal_only_ensemble_is_blind_to_structure(reference: np.ndarray) -> None:
    ens = EnsembleDriftDetector(covariance=False).fit(reference)
    result = ens.detect(np.random.default_rng(10).standard_normal((300, D)))
    assert not result.is_drift
    assert result.details["covariance_result"] is None


def test_factory_builds_covariance_detector() -> None:
    assert isinstance(create_drift_detector("covariance"), CovarianceDriftDetector)


def test_psi_does_not_fire_on_small_drift_free_windows() -> None:
    rng = np.random.default_rng(11)
    det = PopulationStabilityIndexDetector().fit(rng.standard_normal((2000, 4)))
    flags = [det.detect(rng.standard_normal((40, 4))).is_drift for _ in range(40)]
    assert sum(flags) <= 4


def test_psi_detects_a_real_shift() -> None:
    rng = np.random.default_rng(12)
    det = PopulationStabilityIndexDetector().fit(rng.standard_normal((2000, 3)))
    result = det.detect(rng.normal(1.0, 1.0, (400, 3)))
    assert result.is_drift
    assert result.p_value < 1e-6
    assert result.test_statistic >= 0.10


def test_psi_bins_are_open_ended_quantiles() -> None:
    det = PopulationStabilityIndexDetector(n_bins=5).fit(np.arange(1000.0).reshape(-1, 1))
    edges = det.bin_edges[0]
    assert edges[0] == -np.inf and edges[-1] == np.inf
    counts, _ = np.histogram(np.arange(1000.0), bins=edges)
    assert counts.min() >= 190  # roughly equal reference mass per bin


def test_psi_constant_feature_keeps_two_bins() -> None:
    det = PopulationStabilityIndexDetector().fit(np.ones((100, 1)))
    assert len(det.bin_edges[0]) >= 3
    assert not det.detect(np.ones((50, 1))).is_drift
