# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one weighting the fusion routes share.

``POST /api/v1/detect/fusion`` and the batch job's ``fusion`` method score
the same components (statistical and temporal z-scores, with dimensional and
spectral available to the synchronous route) and must reach the same verdict
on the same data. They did not: the synchronous route normalised
0.35/0.25 for statistical+temporal (0.583/0.417) while the batch path hard
coded 0.6/0.4, so eleven zeros followed by a 10 at sensitivity 0.52 was an
anomaly over HTTP (3.442 against 3.44) and clean as a batch job (3.437).
Both now read this table, and ``tests/api/test_fusion_batch_parity.py``
pins the parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = ["FUSION_WEIGHTS", "fuse", "normalized_weights"]

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
