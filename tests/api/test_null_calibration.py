# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The detection thresholds mean what they say at every width and length.

Measured before this calibration, on pure Gaussian noise (600 rows): the
multivariate route flagged 0.3% of rows at width 1, 14% at 8, 72% at 16 and
100% from 32 up, and the fusion route called every noise matrix "anomaly".
Both compared a maximum statistic with a fixed threshold. Every component is
now the z-equivalent of its maximum under the null given the values it ranged
over (``api.fusion_weights``), so the threshold's false-positive rate holds
per row for per-row verdicts and per series for the series verdict, and a
real spike still clears it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from omni_mercury_engine.api import fusion_weights as fw
from omni_mercury_engine.api.routes import detection

pytest.importorskip("fastapi", reason="fastapi (optional '[api]' extra) is required")

from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import app

#: sensitivity 1.0 -> threshold 2.0 -> two-sided tail 4.55%: large enough to
#: measure on a few hundred draws, and the same rule as the default 3.5.
SENSITIVITY = 1.0
ALPHA = 2.0 * norm.sf(2.0)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_extreme_tail_and_calibration_identities() -> None:
    assert fw.extreme_tail(0.01, 1) == pytest.approx(0.01)
    assert fw.extreme_tail(0.01, 100) == pytest.approx(1 - 0.99**100)
    assert fw.extreme_tail(1e-12, 1_000_000) == pytest.approx(1e-6, rel=1e-6)
    assert float(fw.calibrated_max_z(3.5, 1)) == pytest.approx(3.5)
    assert float(fw.chi_norm_z(3.5, 1)) == pytest.approx(3.5)  # one feature: the norm is |z|
    assert float(fw.calibrated_max_z(3.5, 1000)) < 3.5  # a max over many is less surprising
    assert float(fw.chi_norm_z(np.sqrt(64.0), 64)) < 1.0  # an average-length 64-vector is ordinary
    assert float(fw.calibrated_max_z(60.0, 1)) == fw.Z_MAX


@pytest.mark.parametrize("width", [1, 8, 64, 256])
def test_multivariate_false_positive_rate_is_nominal_at_every_width(
    client: TestClient, width: int
) -> None:
    """Per-row flags on pure noise sit at the threshold's tail, not at 100%."""
    flagged = 0
    rows = 0
    for seed in range(3):
        noise = np.random.default_rng(seed).standard_normal((600, width))
        response = client.post(
            "/api/v1/detect/multivariate", json={"data": noise.tolist(), "sensitivity": SENSITIVITY}
        )
        assert response.status_code == 200, response.text
        flagged += sum(response.json()["anomalies"])
        rows += 600
    rate = flagged / rows
    assert 0.5 * ALPHA <= rate <= 2.0 * ALPHA, f"width {width}: {rate:.4f} vs nominal {ALPHA:.4f}"


@pytest.mark.parametrize("width", [1, 8, 64])
def test_fusion_series_false_positive_rate_is_nominal(width: int) -> None:
    """The whole-series verdict on noise is 'anomaly' about as often as the threshold's tail."""
    flags = 0
    draws = 120
    for seed in range(draws):
        noise = np.random.default_rng(1000 + seed).standard_normal((100, width))
        request = detection.FusionRequest(
            data=noise.tolist(),
            detectors=["statistical", "temporal", "dimensional"],
            sensitivity=SENSITIVITY,
        )
        flags += asyncio.run(detection.detect_fusion(request, None)).is_anomaly
    rate = flags / draws
    # A weighted sum of calibrated components is at most their maximum, so the
    # series rate sits at or below the nominal tail; it must not be far above.
    assert rate <= 2.5 * ALPHA, f"width {width}: series false-positive rate {rate:.3f}"


def test_a_spike_is_still_found_at_width_64(client: TestClient) -> None:
    noise = np.random.default_rng(7).standard_normal((600, 64))
    noise[300, 5] += 10.0
    response = client.post(
        "/api/v1/detect/multivariate", json={"data": noise.tolist(), "sensitivity": 0.5}
    )
    flagged = [i for i, f in enumerate(response.json()["anomalies"]) if f]
    assert 300 in flagged and len(flagged) <= 3
    fusion = asyncio.run(
        detection.detect_fusion(detection.FusionRequest(data=noise.tolist(), sensitivity=0.5), None)
    )
    assert fusion.is_anomaly is True


def test_masking_is_gone(client: TestClient) -> None:
    """A spike no longer inflates the spread it is judged against."""
    data = [1.0, 1.1, 1.0, 1.1, 1.0, 100.0, 1.0, 1.1, 1.0, 1.1]
    response = client.post(
        "/api/v1/detect/univariate", json={"data": data, "sensitivity": 0.5, "method": "zscore"}
    )
    body = response.json()
    assert body["anomalies"] == [i == 5 for i in range(10)]
    assert body["summary"]["location"] == pytest.approx(1.05)
    assert body["summary"]["scale"] < 0.1
    series = [0.0] * 11 + [10.0]
    fused = asyncio.run(
        detection.detect_fusion(detection.FusionRequest(data=series, sensitivity=0.52), None)
    )
    assert fused.is_anomaly is True


def _fusion(series: list[Any], **kw: object) -> detection.FusionResponse:
    request = detection.FusionRequest(data=series, **kw)  # type: ignore[arg-type]
    response: detection.FusionResponse = asyncio.run(detection.detect_fusion(request, None))
    return response


def test_a_level_shift_is_a_collective_anomaly() -> None:
    """A 4-sigma shift over the last third: no single row stands out, the stretch does."""
    x = np.random.default_rng(3).standard_normal(300)
    x[200:] += 4.0
    response = _fusion(x.tolist(), sensitivity=0.5)
    assert response.is_anomaly is True
    assert response.detector_scores["collective"] > 3.5
    assert response.detector_scores["statistical"] < 3.5  # the point components see nothing
    matrix = x.reshape(-1, 1)
    location, scale = fw.column_statistics(matrix)
    rows = fw.row_components(matrix, location, scale, dof=fw.null_dof(300))["collective"]
    assert np.mean(rows[200:] > 3.5) > 0.9 and np.mean(rows[:190] > 3.5) < 0.05


def test_a_spike_spread_over_its_neighbours_is_still_found() -> None:
    """An attacker spends an 8-sigma spike as five 3.6-sigma rows; the window sees the sum."""
    x = np.random.default_rng(4).standard_normal(300)
    x[148:153] += 8.0 / np.sqrt(5.0)
    response = _fusion(x.tolist(), sensitivity=0.5)
    assert response.is_anomaly is True and response.detector_scores["collective"] > 3.5
    matrix = x.reshape(-1, 1)
    location, scale = fw.column_statistics(matrix)
    rows = fw.row_components(matrix, location, scale, dof=fw.null_dof(300))["collective"]
    assert np.all(rows[148:153] > 3.5)
    assert int(np.sum(rows > 3.5)) <= 12  # the attribution stays near the attack


def test_a_spike_riding_a_season_is_scored_at_its_true_size(client: TestClient) -> None:
    """One seasonal amplitude above a crest: a z of 2 against the season, 6 against the residual."""
    t = np.arange(300)
    x = 3.0 * np.sin(2 * np.pi * t / 50.0) + 0.5 * np.random.default_rng(5).standard_normal(300)
    crest = int(np.argmax(x[100:200])) + 100
    x[crest] += 3.0
    body = client.post(
        "/api/v1/detect/univariate", json={"data": x.tolist(), "sensitivity": 0.5}
    ).json()
    assert body["anomalies"][crest] is True
    assert sum(body["anomalies"]) <= 3
    assert body["summary"]["seasonal_period"] == 50.0
    multivariate = client.post(
        "/api/v1/detect/multivariate",
        json={"data": x.reshape(-1, 1).tolist(), "sensitivity": 0.5},
    ).json()
    assert multivariate["anomalies"][crest] is True
    assert multivariate["summary"]["seasonal_periods"] == [50]
    fused = _fusion(x.tolist(), sensitivity=0.5)
    assert fused.is_anomaly is True


def test_one_decisive_component_carries_the_verdict() -> None:
    """A pure tone: the spectral z is decisive although the point components see noise."""
    t = np.arange(256)
    tone = np.sin(2 * np.pi * 20 * t / 256) + 0.05 * np.random.default_rng(6).standard_normal(256)
    response = _fusion(
        tone.tolist(), detectors=["statistical", "temporal", "dimensional", "spectral"]
    )
    assert response.detector_scores["spectral"] > 10
    assert response.is_anomaly is True
    assert "decisive" in response.explanation


def test_noise_is_still_clean_with_every_component() -> None:
    flags = 0
    for seed in range(60):
        noise = np.random.default_rng(2000 + seed).standard_normal((120, 4))
        response = _fusion(
            noise.tolist(),
            detectors=["statistical", "temporal", "collective", "dimensional", "spectral"],
            sensitivity=SENSITIVITY,
        )
        flags += response.is_anomaly
    assert flags / 60 <= 2.5 * ALPHA
