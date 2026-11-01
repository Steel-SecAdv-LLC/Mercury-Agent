# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch fusion and synchronous fusion reach the same verdict on the same data.

The synchronous route normalised weights of 0.35/0.25 for statistical and
temporal (0.583/0.417) while the batch path hard coded 0.6/0.4, so eleven
zeros followed by a 10 at sensitivity 0.52 was an anomaly over HTTP (3.442
against a 3.44 threshold) and clean as a batch job (3.437). Both routes now
read ``api.fusion_weights``; these tests pin the table and the parity.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from omni_mercury_engine.api import fusion_weights
from omni_mercury_engine.api.routes import batch, detection

SERIES = [0.0] * 11 + [10.0]
SENSITIVITY = 0.52


def _sync_fused(
    series: list[float] | list[list[float]], detectors: list[str]
) -> tuple[float, bool]:
    request = detection.FusionRequest(data=series, detectors=detectors, sensitivity=SENSITIVITY)
    response = asyncio.run(detection.detect_fusion(request, None))
    return response.fused_score, response.is_anomaly


def _batch_fused(series: list[float] | list[list[float]]) -> tuple[float, bool]:
    rows = [[v] for v in series] if not isinstance(series[0], list) else [list(r) for r in series]
    arr = np.asarray(rows, dtype=float)
    results = asyncio.run(
        batch._process_chunk(
            rows,
            batch.BatchDetectionMethod.FUSION,
            SENSITIVITY,
            0,
            np.mean(arr, axis=0),
            np.std(arr, axis=0) + 1e-8,
            None,
        )
    )
    peak = max(results, key=lambda r: r["score"])
    return float(peak["score"]), bool(peak["is_anomaly"])


def test_weights_normalise_over_the_selected_components() -> None:
    both = fusion_weights.normalized_weights(["statistical", "temporal"])
    assert both["statistical"] == pytest.approx(0.35 / 0.60)
    assert both["temporal"] == pytest.approx(0.25 / 0.60)
    assert sum(
        fusion_weights.normalized_weights(fusion_weights.FUSION_WEIGHTS).values()
    ) == pytest.approx(1.0)
    assert fusion_weights.normalized_weights(["unknown"]) == {}
    assert fusion_weights.fuse({"statistical": 3.0}, ["unknown"]) == 0.0


def test_batch_fusion_reaches_the_synchronous_verdict() -> None:
    """The reviewer's series: anomalous over HTTP, it must be anomalous as a batch job too."""
    sync_score, sync_flag = _sync_fused(SERIES, ["statistical", "temporal"])
    batch_score, batch_flag = _batch_fused(SERIES)
    assert sync_flag is True
    assert batch_flag is sync_flag
    assert batch_score == pytest.approx(sync_score, abs=1e-6)


def test_isolated_spike_scores_identically_on_both_routes() -> None:
    """Rise and fall of equal size: the spike sample carries both series maxima."""
    series = [0.0] * 10 + [10.0] + [0.0] * 10
    sync_score, sync_flag = _sync_fused(series, ["statistical", "temporal"])
    batch_score, batch_flag = _batch_fused(series)
    assert batch_score == pytest.approx(sync_score, abs=1e-6)
    assert batch_flag is sync_flag


@pytest.mark.parametrize("seed", range(5))
def test_batch_peak_never_exceeds_the_synchronous_score(seed: int) -> None:
    """Per-sample components are bounded by the series maxima the synchronous route fuses.

    A sample's statistical component is at most the series' largest |z| and
    its temporal component at most the largest |step|, so with the shared
    weights the batch peak can only fall short of the synchronous score when
    the two maxima sit on different samples; it can never exceed it, and a
    batch flag therefore always implies a synchronous flag.
    """
    rng = np.random.default_rng(seed)
    series = list(rng.standard_normal(64))
    series[int(rng.integers(1, 63))] += float(rng.uniform(4.0, 9.0))
    sync_score, sync_flag = _sync_fused(series, ["statistical", "temporal"])
    batch_score, batch_flag = _batch_fused(series)
    # The batch route rounds each sample's score to six decimals.
    assert batch_score <= sync_score + 5e-7
    assert not batch_flag or sync_flag


def test_synchronous_route_reports_the_shared_weights() -> None:
    request = detection.FusionRequest(
        data=SERIES, detectors=["statistical", "temporal"], sensitivity=SENSITIVITY
    )
    response = asyncio.run(detection.detect_fusion(request, None))
    assert response.detector_contributions == pytest.approx(
        fusion_weights.normalized_weights(["statistical", "temporal"])
    )


def test_two_column_input_reaches_the_same_verdict_on_both_routes() -> None:
    """The reviewer's matrix: a 10-step in one column beside a constant 1000 in the other.

    The synchronous route standardised against one global mean and std and
    differenced the flattened matrix (a "step" ran from 1000 to 0 across the
    row boundary), scoring 1.418 and clean; the batch job standardised per
    column and differenced row to row, scoring 3.442 and anomalous. Both now
    standardise per feature and step row to row.
    """
    matrix = [[0.0, 1000.0]] * 11 + [[10.0, 1000.0]]
    sync_score, sync_flag = _sync_fused(matrix, ["statistical", "temporal"])
    batch_score, batch_flag = _batch_fused(matrix)
    assert sync_flag is True and batch_flag is True
    assert batch_score == pytest.approx(sync_score, abs=5e-7)


def test_the_constant_feature_does_not_dominate_the_fused_score() -> None:
    """A constant column scores 0 on every component rather than inf or a false step."""
    matrix = [[v, 1000.0] for v in [0.0] * 10 + [10.0] + [0.0] * 10]
    fused_two, _ = _sync_fused(matrix, ["statistical", "temporal"])
    fused_one, _ = _sync_fused([0.0] * 10 + [10.0] + [0.0] * 10, ["statistical", "temporal"])
    assert fused_two == pytest.approx(fused_one, abs=1e-9)


def test_spectral_component_is_taken_per_feature() -> None:
    """A tone in one column is found whether or not a constant column sits beside it."""
    t = np.arange(256)
    tone = list(np.sin(2 * np.pi * 20 * t / 256))
    alone = asyncio.run(
        detection.detect_fusion(detection.FusionRequest(data=tone, detectors=["spectral"]), None)
    )
    beside = asyncio.run(
        detection.detect_fusion(
            detection.FusionRequest(data=[[v, 5.0] for v in tone], detectors=["spectral"]), None
        )
    )
    assert beside.detector_scores["spectral"] == pytest.approx(alone.detector_scores["spectral"])
