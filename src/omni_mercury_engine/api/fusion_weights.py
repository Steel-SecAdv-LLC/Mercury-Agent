# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The scoring kernel the fusion routes share: components and weights.

``POST /api/v1/detect/fusion`` and the batch job's ``fusion`` method score
the same components (statistical and temporal z-scores, with dimensional and
spectral available to the synchronous route) and must reach the same verdict
on the same data. They did not, twice over. The synchronous route normalised
0.35/0.25 for statistical+temporal (0.583/0.417) while the batch path hard
coded 0.6/0.4, so eleven zeros followed by a 10 at sensitivity 0.52 was an
anomaly over HTTP and clean as a batch job. And on multi-feature input the
synchronous route standardised every value against one global mean and
standard deviation and differenced the *flattened* matrix (so a "step" ran
from the last column of one row to the first column of the next), while the
batch path standardised per column and differenced row to row; for
``[[0, 1000]] * 11 + [[10, 1000]]`` the batch job flagged (3.442) and the
synchronous route did not (1.418).

Both routes now take their components from here: :func:`as_matrix` shapes
the input as rows of features (a 1-D series is one feature),
:func:`column_statistics` standardises per feature, :func:`statistical_scores`
and :func:`temporal_scores` give each row its largest per-feature |z| and its
largest per-feature step from the previous row in standard-deviation units,
and :func:`fuse` applies :data:`FUSION_WEIGHTS` normalised over the selected
components. ``tests/api/test_fusion_batch_parity.py`` pins both parities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "EPSILON",
    "FUSION_WEIGHTS",
    "as_matrix",
    "column_statistics",
    "fuse",
    "normalized_weights",
    "statistical_scores",
    "temporal_scores",
]

#: Added to every standard deviation so a constant feature scores 0, not inf.
EPSILON = 1e-8

#: Relative weight of each component before normalisation over the components
#: a request selects. The order is the documented one.
FUSION_WEIGHTS: Mapping[str, float] = {
    "statistical": 0.35,
    "temporal": 0.25,
    "dimensional": 0.20,
    "spectral": 0.20,
}


def normalized_weights(detectors: Iterable[str]) -> dict[str, float]:
    """Weights of the selected components, scaled to sum to one.

    Unknown names are ignored (the routes validate their own detector
    lists); an empty or wholly unknown selection yields an empty mapping, so
    a fused score over it is ``0.0`` rather than a division by zero.
    """
    selected = {d: FUSION_WEIGHTS[d] for d in detectors if d in FUSION_WEIGHTS}
    total = sum(selected.values())
    if total <= 0.0:
        return {}
    return {d: w / total for d, w in selected.items()}


def fuse(scores: Mapping[str, float], detectors: Iterable[str]) -> float:
    """The weighted sum of ``scores`` over ``detectors`` with the shared weights."""
    weights = normalized_weights(detectors)
    return float(sum(scores.get(d, 0.0) * w for d, w in weights.items()))


def as_matrix(data: Any) -> np.ndarray[Any, Any]:
    """``data`` as rows of features: a 1-D series becomes one feature column."""
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"fusion input must be 1-D or 2-D, got {matrix.ndim}-D")
    return matrix


def column_statistics(
    matrix: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Per-feature mean and standard deviation (plus :data:`EPSILON`) over the rows."""
    return np.mean(matrix, axis=0), np.std(matrix, axis=0) + EPSILON


def statistical_scores(
    matrix: np.ndarray[Any, Any], mean: np.ndarray[Any, Any], std: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Each row's largest per-feature |z| against ``mean`` and ``std``."""
    z = np.abs((matrix - mean) / std)
    return np.max(z, axis=1)


def temporal_scores(
    matrix: np.ndarray[Any, Any],
    std: np.ndarray[Any, Any],
    previous: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Each row's largest per-feature step from the row before it, in units of ``std``.

    The first row scores 0 unless ``previous`` (the row before this block,
    for chunked scoring) is given. Steps are taken row to row within each
    feature, never across the flattened matrix.
    """
    if matrix.shape[0] == 0:
        return np.zeros(0)
    if previous is None:
        starts = np.vstack([matrix[:1], matrix[:-1]])
        steps = np.max(np.abs(matrix - starts) / std, axis=1)
        steps[0] = 0.0
        return steps
    starts = np.vstack([np.asarray(previous, dtype=float).reshape(1, -1), matrix[:-1]])
    return np.max(np.abs(matrix - starts) / std, axis=1)
