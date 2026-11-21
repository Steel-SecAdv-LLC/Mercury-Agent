# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The entropy-constrained R/K/IG weight re-selection, pinned.

``research/ensembles/harness_weight_entropy.py`` swept the whole weight
simplex (step 0.05) over the detector's own component scores on the recorded
tables, with a normalised-entropy band [0.30, 0.95] excluding both the
collapse onto a single component and the diffuse near-uniform extreme.
Measured: the in-band winner w = (0.65, 0.15, 0.20) (H = 0.807) reaches mean
AUROC 0.8922 against the shipped adaptive weights' 0.8897 and the 40/30/30
prior's 0.8810.

The winner is *recorded, not adopted*: it was selected on the same five
tables it is scored on, and a +0.0025 in-sample margin is selection bias
territory — adopting it would be the exact bar-lowering move the repository's
own adoption bars exist to refuse. What ships from this item is the
constraint machinery and the measured surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.ensembles.harness_weight_entropy import (
    ENTROPY_BAND,
    normalized_entropy,
    simplex_grid,
)

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "fusion_weight_entropy.json"


def test_normalized_entropy_spans_the_unit_interval() -> None:
    assert normalized_entropy(np.array([1.0, 0.0, 0.0])) == 0.0
    assert normalized_entropy(np.array([1 / 3, 1 / 3, 1 / 3])) == pytest.approx(1.0)
    middling = normalized_entropy(np.array([0.65, 0.15, 0.20]))
    assert 0.0 < middling < 1.0


def test_the_grid_is_the_whole_simplex() -> None:
    grid = simplex_grid()
    assert len(grid) == 231  # C(22, 2): the 0.05-step 3-simplex
    for w in grid:
        assert w.min() >= -1e-9
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_the_band_excludes_both_extremes() -> None:
    low, high = ENTROPY_BAND
    assert normalized_entropy(np.array([1.0, 0.0, 0.0])) < low
    assert normalized_entropy(np.array([1 / 3, 1 / 3, 1 / 3])) > high


def test_the_artifact_records_the_selection_and_its_nonadoption() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "mercury.research.fusion_weight_entropy/v1"
    selected = artifact["selected_in_band"]
    low, high = artifact["config"]["entropy_band"]
    assert (
        low <= selected["normalized_entropy"] <= high
    ), "the selected configuration must honour its own entropy constraint"
    shipped = artifact["references"]["shipped_adaptive"]
    assert selected["mean_rank"] < shipped["mean_rank"]
    # The margin over the shipped adaptive weights is small and in-sample;
    # if it ever grows past this ceiling, re-examine adoption properly
    # (held-out selection, the full operational metrics) instead of here.
    assert selected["mean_auc"] - shipped["mean_auc"] <= 0.02


def test_the_shipped_adaptive_default_was_not_silently_replaced() -> None:
    """Adoption needs held-out evidence; the default weighting stays."""
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    detector = MercuryAnomalyDetector()
    base = detector._data_type_default_weights()
    np.testing.assert_allclose(base, [0.4, 0.3, 0.3], atol=1e-9)
