# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch fusion and synchronous fusion score the same data with one kernel.

The synchronous route once normalised weights of 0.35/0.25 for statistical
and temporal while the batch path hard coded 0.6/0.4, standardised against one
global mean and differenced the flattened matrix. Both now read
``api.fusion_weights``: the same robust standardisation (median, scaled MAD),
the same per-feature components, the same weights, and the same null
calibration. The batch job gives each row a verdict (its maxima calibrated to
the row's width); the synchronous route gives the whole series one verdict
(its maxima calibrated to every value of the series). These tests pin the
kernel, the parity between the batch job and the kernel's per-row scores, and
the relation between the two verdicts.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.api import fusion_weights
from omni_mercury_engine.api.routes import batch, detection

SERIES = [0.0] * 11 + [10.0]
SENSITIVITY = 0.52
THRESHOLD = 2.0 + (1.0 - SENSITIVITY) * 3.0


def _rows(series: list[float] | list[list[float]]) -> list[list[float]]:
    return [[v] for v in series] if not isinstance(series[0], list) else [list(r) for r in series]


def _sync(
    series: list[float] | list[list[float]], detectors: list[str]
) -> detection.FusionResponse:
    request = detection.FusionRequest(data=series, detectors=detectors, sensitivity=SENSITIVITY)
    response: detection.FusionResponse = asyncio.run(detection.detect_fusion(request, None))
    return response


def _batch(series: list[float] | list[list[float]]) -> list[dict[str, Any]]:
    """Score ``series`` the way the batch job does: whole-data-set context, then chunks."""
    rows = _rows(series)
    matrix = np.asarray(rows, dtype=float)
    residual, _ = fusion_weights.deseasonalize(matrix)
    location, scale = fusion_weights.column_statistics(residual)
    dof = fusion_weights.null_dof(len(rows))
    collective = fusion_weights.row_components(residual, location, scale, dof=dof)["collective"]
    return asyncio.run(
        batch._process_chunk(
            rows,
            batch.BatchDetectionMethod.FUSION,
            SENSITIVITY,
            0,
            location,
            scale,
            None,
            dof,
            baseline=matrix - residual,
            collective_rows=collective,
        )
    )


def _kernel_rows(series: list[float] | list[list[float]]) -> list[float]:
    """What the batch job must produce: the kernel's per-row components, fused."""
    matrix = np.asarray(_rows(series), dtype=float)
    residual, _ = fusion_weights.deseasonalize(matrix)
    location, scale = fusion_weights.column_statistics(residual)
    parts = fusion_weights.row_components(
        residual, location, scale, dof=fusion_weights.null_dof(matrix.shape[0])
    )
    return [
        fusion_weights.fuse_calibrated(
            {
                "statistical": float(parts["statistical"][i]),
                "temporal": float(parts["temporal"][i]),
                "collective": float(parts["collective"][i]),
            },
            batch._BATCH_FUSION_COMPONENTS,
        )
        for i in range(matrix.shape[0])
    ]


def test_weights_normalise_over_the_selected_components() -> None:
    both = fusion_weights.normalized_weights(["statistical", "temporal"])
    assert both["statistical"] == pytest.approx(0.30 / 0.50)
    assert both["temporal"] == pytest.approx(0.20 / 0.50)
    assert sum(
        fusion_weights.normalized_weights(fusion_weights.FUSION_WEIGHTS).values()
    ) == pytest.approx(1.0)
    assert fusion_weights.normalized_weights(["unknown"]) == {}
    assert fusion_weights.fuse({"statistical": 3.0}, ["unknown"]) == 0.0


@pytest.mark.parametrize(
    "series",
    [SERIES, [0.0] * 10 + [10.0] + [0.0] * 10, [[0.0, 1000.0]] * 11 + [[10.0, 1000.0]]],
    ids=["reviewer-series", "isolated-spike", "two-columns"],
)
def test_batch_rows_are_the_kernels_rows(series: list[Any]) -> None:
    """Every batch sample score is the kernel's fused per-row score (to the job's rounding)."""
    scored = _batch(series)
    expected = _kernel_rows(series)
    assert [r["score"] for r in scored] == pytest.approx(expected, abs=1e-6)
    assert [r["is_anomaly"] for r in scored] == [v > THRESHOLD for v in expected]


@pytest.mark.parametrize(
    "series",
    [SERIES, [0.0] * 10 + [10.0] + [0.0] * 10, [[0.0, 1000.0]] * 11 + [[10.0, 1000.0]]],
    ids=["reviewer-series", "isolated-spike", "two-columns"],
)
def test_both_routes_flag_a_spike_the_spread_cannot_hide(series: list[Any]) -> None:
    """The reviewer's cases: a 10 among zeros is anomalous on both routes.

    With the mean and standard deviation the spike inflated its own
    denominator to a z of 3.3; against the median and MAD it is unmistakable.
    """
    response = _sync(series, ["statistical", "temporal"])
    peak = max(_batch(series), key=lambda r: r["score"])
    assert response.is_anomaly is True
    assert peak["is_anomaly"] is True
    assert peak["index"] == len(series) - 1 if series is SERIES else True


def test_synchronous_score_is_the_series_calibrated_fusion() -> None:
    """The route's fused score is the kernel's whole-series components, fused."""
    matrix = np.asarray(_rows(SERIES), dtype=float)
    location, scale = fusion_weights.column_statistics(matrix)
    expected = fusion_weights.fuse_calibrated(
        fusion_weights.series_components(
            matrix, location, scale, dof=fusion_weights.null_dof(matrix.shape[0])
        ),
        ["statistical", "temporal"],
    )
    response = _sync(SERIES, ["statistical", "temporal"])
    assert response.fused_score == pytest.approx(expected, abs=1e-9)
    assert response.detector_contributions == pytest.approx(
        fusion_weights.normalized_weights(["statistical", "temporal"])
    )


@pytest.mark.parametrize("seed", range(5))
def test_series_verdict_never_exceeds_the_rows_evidence(seed: int) -> None:
    """A whole-series component is at most the largest per-row component.

    The series maximum ranges over every row, so its calibrated z can only be
    smaller than the same statistic calibrated to one row's width: more
    comparisons make the same extreme less surprising.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((64, 3))
    matrix[int(rng.integers(1, 63)), 1] += float(rng.uniform(4.0, 9.0))
    location, scale = fusion_weights.column_statistics(matrix)
    dof = fusion_weights.null_dof(matrix.shape[0])
    series = fusion_weights.series_components(matrix, location, scale, dof=dof)
    rows = fusion_weights.row_components(matrix, location, scale, dof=dof)
    for name in ("statistical", "temporal", "dimensional", "collective"):
        assert series[name] <= float(np.max(rows[name])) + 1e-9


def test_a_constant_feature_adds_no_evidence() -> None:
    """A constant column scores 0 on every component and cannot raise the fused score."""
    spike = [0.0] * 10 + [10.0] + [0.0] * 10
    with_constant = _sync([[v, 1000.0] for v in spike], ["statistical", "temporal"])
    alone = _sync(spike, ["statistical", "temporal"])
    assert with_constant.is_anomaly is True and alone.is_anomaly is True
    assert with_constant.fused_score <= alone.fused_score + 1e-9
    # It is one more comparison per row, nothing else: doubling the values a
    # maximum ranges over halves its rarity, about 0.12 in z at this level.
    assert alone.fused_score - with_constant.fused_score < 0.25


def test_spectral_component_is_taken_per_feature() -> None:
    """A tone in one column is found whether or not a constant column sits beside it."""
    t = np.arange(256)
    tone = list(np.sin(2 * np.pi * 20 * t / 256))
    alone = _sync(tone, ["spectral"])
    beside = _sync([[v, 5.0] for v in tone], ["spectral"])
    assert alone.is_anomaly is True and beside.is_anomaly is True
    # Two features: the peak is the larger of two, calibrated as such.
    assert beside.detector_scores["spectral"] == pytest.approx(
        float(fusion_weights.calibrated_max_z(alone.detector_scores["spectral"], 2)), abs=1e-6
    )
