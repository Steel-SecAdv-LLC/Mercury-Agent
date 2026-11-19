# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contrastive probe refinement: mechanism and measured verdict, pinned.

``AnomalyMathArrest.refine_contrastive`` re-weights probes through the
decorrelator's multipliers using the fused score's own discordant
(anomaly, normal) pairs on a labelled calibration fold — a probe-local
adjustment through the ensemble's existing weighting seam, never a change to
any probe's mathematics. Measured on the five recorded tables x three seeds
(``research/math_arrest/harness_contrastive_refinement.py``): held-out mean
AUROC 0.6643 unrefined → 0.6799 / 0.6930 / 0.7114 at eta 0.25 / 0.5 / 1.0,
mean rank improving at every eta, calibration discordance 0.313 → 0.279.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.xdist_group("contrastive_refinement"),
    pytest.mark.timeout(600),
]

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.math_arrest.fusion import CorrelationAwareDecorrelator
from tests.recorded_data import adbench, stratified_split

ARTIFACT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "math_arrest_contrastive_refinement.json"
)


@pytest.fixture(scope="module")
def folds() -> dict[str, np.ndarray]:
    features, labels = adbench("breastw")
    train_all, test = stratified_split(labels, 0)
    fit_local, cal_local = stratified_split(labels[train_all], 0)
    fit_rows, cal_rows = train_all[fit_local], train_all[cal_local]
    mean = features[fit_rows].mean(axis=0)
    spread = features[fit_rows].std(axis=0)
    spread[spread < 1e-8] = 1.0
    return {
        "fit": (features[fit_rows] - mean) / spread,
        "cal": (features[cal_rows] - mean) / spread,
        "cal_y": labels[cal_rows],
    }


def test_refinement_reduces_the_calibration_discordance(folds: dict[str, np.ndarray]) -> None:
    detector = AnomalyMathArrest()
    detector.fit(folds["fit"])
    report = detector.refine_contrastive(folds["cal"], folds["cal_y"], eta=0.5)
    assert report["discordant_after"] <= report["discordant_before"], (
        f"refinement raised discordance: {report['discordant_before']:.4f} -> "
        f"{report['discordant_after']:.4f}"
    )
    for value in report["multipliers"].values():
        assert 0.0 < value <= 1.0


def test_refinement_never_touches_the_probe_mathematics(folds: dict[str, np.ndarray]) -> None:
    detector = AnomalyMathArrest()
    detector.fit(folds["fit"])
    before = {k: v.copy() for k, v in detector.per_probe_scores(folds["cal"]).items()}
    detector.refine_contrastive(folds["cal"], folds["cal_y"], eta=1.0)
    after = detector.per_probe_scores(folds["cal"])
    for name, column in before.items():
        np.testing.assert_array_equal(column, after[name])


def test_eta_zero_is_a_no_op_on_the_multipliers(folds: dict[str, np.ndarray]) -> None:
    detector = AnomalyMathArrest()
    detector.fit(folds["fit"])
    before = detector._decorrelator.weight_multipliers
    report = detector.refine_contrastive(folds["cal"], folds["cal_y"], eta=0.0)
    assert report["multipliers"] == pytest.approx(before)


def test_inputs_are_validated(folds: dict[str, np.ndarray]) -> None:
    detector = AnomalyMathArrest()
    detector.fit(folds["fit"])
    with pytest.raises(ValueError, match="eta must be"):
        detector.refine_contrastive(folds["cal"], folds["cal_y"], eta=-0.1)
    with pytest.raises(ValueError, match="both classes"):
        detector.refine_contrastive(folds["cal"], np.zeros(len(folds["cal"]), dtype=np.int64))
    decorrelator = CorrelationAwareDecorrelator()
    with pytest.raises(RuntimeError, match="has not been calibrated"):
        decorrelator.scale_multipliers({"a": 1.0})


def test_the_artifact_pins_the_measured_improvement() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "mercury.research.math_arrest_contrastive_refinement/v1"
    results = artifact["results"]
    baseline = results["eta_0.0"]
    # Every measured eta improved the mean rank; the strongest gain is pinned
    # with a floor under the measured +0.0471 of mean AUROC.
    assert set(artifact["improving_etas"]) >= {"eta_0.5", "eta_1.0"}
    assert results["eta_1.0"]["mean_auc"] >= baseline["mean_auc"] + 0.01
    assert results["eta_1.0"]["mean_rank"] < baseline["mean_rank"]
    disc = artifact["calibration_discordance"]
    assert disc["mean_after_eta_0.5"] <= disc["mean_before"]
