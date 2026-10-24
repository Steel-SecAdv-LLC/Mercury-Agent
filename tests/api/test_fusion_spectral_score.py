# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The fusion route's spectral score is in z-score units, like the others.

It was ``max|FFT| / N`` of the raw data: its DC bin alone is the mean, so the
same signal offset by 1000 scored 1000 and scaled by 0.001 scored 0.00015,
while the statistical score stayed 3.07; any data away from zero was an
anomaly. It is now Fisher's test for a dominant frequency, as a z-score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.stats import norm

from omni_mercury_engine.api.routes import detection
from omni_mercury_engine.api.routes.detection import _spectral_peak_z


def _noise(seed: int, n: int = 256) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


@pytest.mark.parametrize(("scale", "offset"), [(1.0, 1000.0), (1000.0, 0.0), (0.001, -5.0)])
def test_the_score_is_invariant_to_offset_and_scale(scale: float, offset: float) -> None:
    x = _noise(0)
    assert _spectral_peak_z(x * scale + offset) == pytest.approx(_spectral_peak_z(x), rel=1e-6)


def test_a_periodic_component_scores_high_and_noise_does_not() -> None:
    t = np.arange(256)
    tone = np.sin(2 * np.pi * 20 * t / 256)
    assert _spectral_peak_z(_noise(1)) < 2.0
    assert _spectral_peak_z(_noise(1) + 0.5 * tone) > 4.0
    assert _spectral_peak_z(tone) > 10.0


def test_white_noise_false_alarms_do_not_exceed_the_nominal_tail() -> None:
    """Calibrated: the z-score is a normal quantile of a p-value, so at z > t the
    false-alarm rate on white noise is at most the two-sided normal tail."""
    z = np.array([_spectral_peak_z(_noise(seed)) for seed in range(3000)])
    for threshold in (2.0, 2.5, 3.0):
        nominal = 2.0 * norm.sf(threshold)
        assert np.mean(z > threshold) <= nominal * 1.15  # sampling margin on 3,000 draws


@pytest.mark.parametrize("values", [np.ones(50), np.array([1.0, 2.0]), np.zeros(0)])
def test_degenerate_series_score_zero(values: np.ndarray) -> None:
    assert _spectral_peak_z(values) == 0.0


def test_the_route_scores_an_offset_series_like_the_same_series_without_one() -> None:
    """Before: the offset series' fused score was ~1000x and always an anomaly."""
    app = FastAPI()
    app.include_router(detection.router)
    client = TestClient(app)
    x = _noise(2, 64)

    def fused(data: np.ndarray) -> dict[str, Any]:
        response = client.post(
            "/api/v1/detect/fusion",
            json={"data": data.tolist(), "detectors": ["statistical", "spectral"]},
        )
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        return body

    plain, offset = fused(x), fused(x + 1000.0)
    assert offset["fused_score"] == pytest.approx(plain["fused_score"], rel=1e-6)
    assert offset["is_anomaly"] == plain["is_anomaly"]
    assert plain["is_anomaly"] is False
