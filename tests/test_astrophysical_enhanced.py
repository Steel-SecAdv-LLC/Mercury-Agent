# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for enhanced astrophysical anomaly detection model."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.models.astrophysical import AstrophysicalAnomalyModel


def test_astrophysical_black_hole_features() -> None:
    """Test black hole feature extraction"""
    model = AstrophysicalAnomalyModel()
    data = np.random.randn(8, 12)
    features = model.extract_features(data)

    assert features.shape[0] == 8
    assert features.shape[1] == 24


def test_fitted_deviation_flags_departures_in_both_directions() -> None:
    """The fitted reference replaces the inverted proximity heuristic.

    The retired score graded proximity-to-horizon (1/norm), so high-energy
    bursts ranked as maximally safe (measured AUROC 0.000 against clean
    windows, 2026-10-27). Under the fitted reference, rows collapsed toward
    the origin AND high-energy rows must both outscore reference-typical
    rows - direction-free by construction.
    """
    rng = np.random.default_rng(7)
    model = AstrophysicalAnomalyModel().fit(rng.normal(size=(64, 10)))
    typical = rng.normal(size=(8, 10))
    collapsed = rng.normal(size=(8, 10)) * 0.01
    burst = rng.normal(size=(8, 10)) + 12.0

    def mean_score(rows: np.ndarray) -> float:
        return float(np.mean(model.predict(rows)["anomaly_scores"]))

    assert mean_score(collapsed) > mean_score(typical)
    assert mean_score(burst) > mean_score(typical)


def test_unfitted_predict_is_refused() -> None:
    """The retired fixed-direction heuristic is never served as a fallback."""
    import pytest

    with pytest.raises(RuntimeError, match="unfitted"):
        AstrophysicalAnomalyModel().predict(np.random.randn(4, 10))


def test_gravitational_field_computation() -> None:
    """Test gravitational field strength computation"""
    model = AstrophysicalAnomalyModel()

    data = np.random.randn(5, 8)
    features = model.extract_features(data)

    grav_fields = features[:, 1]
    assert np.all(grav_fields >= 0)


def test_omni_code_integration() -> None:
    """Test Omni-Code integration in astrophysical model"""
    model = AstrophysicalAnomalyModel().fit(np.random.randn(16, 6))
    data = np.random.randn(2, 6)

    result = model.predict(data)
    assert "anomaly_scores" in result
