# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fusion monotonicity and per-sample probe attribution, pinned.

Two instruments added to the arrest fusion path:

* **Monotonicity projection** (``PhiWeightedFusion.effective_weights``):
  weights are projected onto the non-negative orthant before normalisation,
  so the fused score is provably non-decreasing in every retained probe's
  score. Every shipped branch already produced non-negative weights, but
  nothing enforced it: a corrupted checkpoint restoring a *negative*
  decorrelator multiplier used to invert that probe's contribution — a rising
  probe score lowered the fused score. That failure now cannot happen, and
  the end-to-end property is validated on the recorded corpus.

* **Leave-one-probe-out attribution**
  (``AnomalyMathArrest.probe_attributions``, persisted in the detector's
  ``detect`` result): per sample, each probe's attribution is the fused-score
  change when that probe is ablated at fixed weight proportions — under the
  shipped ``uniform`` weighting, exactly the fusion of the remaining probes.
  The weight-consistency identity ``sum_j (1 − w_j) · raw_attr_j == 0`` and
  equality with a literal re-fusion are both held here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [
    pytest.mark.xdist_group("fusion_monotonicity"),
    pytest.mark.timeout(600),
]

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.math_arrest.base_probe import ProbeResult
from omni_mercury_engine.detectors.math_arrest.fusion import (
    CorrelationAwareDecorrelator,
    PhiWeightedFusion,
)
from tests.recorded_data import nab_corpus


def _results(scores: list[np.ndarray]) -> list[ProbeResult]:
    return [
        ProbeResult(
            probe_name=f"p{i}",
            deviation_scores=np.asarray(s, dtype=np.float64),
            confidence=1.0,
            trajectory_fit_quality=1.0,
            anomaly_geometry="point",
        )
        for i, s in enumerate(scores)
    ]


def test_a_negative_multiplier_can_no_longer_invert_a_probe() -> None:
    """The projection closes the corrupted-checkpoint inversion.

    Before it, a negative decorrelator multiplier gave probe ``p1`` a
    negative fused weight: raising its score LOWERED the fused score. This
    test fails on that code.
    """
    low = _results([np.full(6, 0.5), np.full(6, 0.2)])
    high = _results([np.full(6, 0.5), np.full(6, 0.9)])  # p1's score rises
    decorrelator = CorrelationAwareDecorrelator()
    decorrelator._weight_multipliers = {"p0": 1.0, "p1": -0.5}  # corrupted state
    decorrelator._is_calibrated = True
    fusion = PhiWeightedFusion(n_probes=2, weighting="uniform")
    fused_low = fusion.fuse(low, decorrelator=decorrelator)
    fused_high = fusion.fuse(high, decorrelator=decorrelator)
    assert np.all(fused_high >= fused_low), (
        "raising a probe's score lowered the fused score: a negative weight "
        "survived the monotonicity projection"
    )
    _, weights = fusion.effective_weights(high, decorrelator=decorrelator)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("weighting", ("uniform", "confidence", "phi_affinity"))
def test_the_fused_score_never_decreases_when_a_probe_score_rises(weighting: str) -> None:
    """The end-to-end property, on every weighting, against random states."""
    rng = np.random.default_rng(0)
    fusion = PhiWeightedFusion(n_probes=5, weighting=weighting)
    for trial in range(25):
        base = [rng.random(8) for _ in range(5)]
        results = _results(base)
        fused = fusion.fuse(results)
        bumped_index = trial % 5
        bumped_scores = [s.copy() for s in base]
        bumped_scores[bumped_index] = np.clip(bumped_scores[bumped_index] + 0.3, 0.0, 1.0)
        fused_bumped = fusion.fuse(_results(bumped_scores))
        assert np.all(fused_bumped >= fused - 1e-12), f"{weighting}, trial {trial}"


def test_monotonicity_holds_on_the_recorded_corpus() -> None:
    """Validated on the independent-label harness: bump any probe, score rises."""
    series = nab_corpus()[0]
    split = series.n_probation
    detector = AnomalyMathArrest()
    detector.fit(series.values[:split])
    held = series.values[split : split + 400]

    results = [p.deviation_score(held) for p in detector._probes if p.is_fitted]
    from omni_mercury_engine.detectors.math_arrest.domain_affinity import get_affinity_order

    order = get_affinity_order("default", [r.probe_name for r in results])
    fused = detector._fusion.fuse(
        results, affinity_order=order, decorrelator=detector._decorrelator
    )
    rng = np.random.default_rng(1)
    for _ in range(5):
        j = int(rng.integers(len(results)))
        bumped = list(results)
        bumped[j] = ProbeResult(
            probe_name=results[j].probe_name,
            deviation_scores=np.clip(results[j].deviation_scores + 0.2, 0.0, 1.0),
            confidence=results[j].confidence,
            trajectory_fit_quality=results[j].trajectory_fit_quality,
            anomaly_geometry=results[j].anomaly_geometry,
        )
        fused_bumped = detector._fusion.fuse(
            bumped, affinity_order=order, decorrelator=detector._decorrelator
        )
        width = min(len(fused), len(fused_bumped))
        assert np.all(fused_bumped[:width] >= fused[:width] - 1e-12)


class TestAttribution:
    def test_attribution_equals_a_literal_refusion_without_the_probe(self) -> None:
        """Under uniform weighting the ablation is exactly the reduced fusion."""
        rng = np.random.default_rng(2)
        results = _results([rng.random(10) for _ in range(4)])
        fusion = PhiWeightedFusion(n_probes=4, weighting="uniform")
        ordered, weights = fusion.effective_weights(results)
        attributions = fusion.leave_one_out_attribution(ordered, weights)
        fused = fusion.fuse(results)
        for j, result in enumerate(ordered):
            reduced = [r for k, r in enumerate(ordered) if k != j]
            refused = PhiWeightedFusion(n_probes=3, weighting="uniform").fuse(reduced)
            np.testing.assert_allclose(attributions[result.probe_name], fused - refused, atol=1e-12)

    def test_the_weight_consistency_identity_holds(self) -> None:
        """sum_j (1 - w_j) * raw_attr_j == 0, tying attributions to weights."""
        rng = np.random.default_rng(3)
        results = _results([0.2 + 0.6 * rng.random(12) for _ in range(5)])
        fusion = PhiWeightedFusion(n_probes=5, weighting="confidence")
        ordered, weights = fusion.effective_weights(results)
        attributions = fusion.leave_one_out_attribution(ordered, weights)
        # Interior scores (no clipping active), so the pre-clip identity is
        # exact on the returned values.
        total = np.zeros(12)
        for j, result in enumerate(ordered):
            total += (1.0 - weights[j]) * attributions[result.probe_name]
        np.testing.assert_allclose(total, 0.0, atol=1e-10)

    def test_the_detector_persists_the_attribution_vector(self) -> None:
        """Every detection carries the per-probe attribution alongside it."""
        from omni_mercury_engine.detectors import MathArrestDetector

        rng = np.random.default_rng(4)
        train = np.sin(np.linspace(0.0, 30.0, 300)) + 0.05 * rng.standard_normal(300)
        detector = MathArrestDetector().fit(train)
        result = detector.detect(train)
        attributions = result["probe_attributions"]
        assert set(attributions) == set(result["probe_names"])
        for column in attributions.values():
            assert column.shape == result["scores"].shape
        # The attributions explain the served score: reconstruct the fused
        # score from the effective weights and the probe scores.
        arrest = detector._arrest
        results = [p.deviation_score(train) for p in arrest._probes if p.is_fitted]
        ordered, weights = arrest._fusion.effective_weights(
            results, decorrelator=arrest._decorrelator
        )
        matrix = np.column_stack([r.deviation_scores for r in ordered])
        reconstructed = np.clip(matrix @ weights, 0.0, 1.0)
        np.testing.assert_allclose(
            reconstructed, np.asarray(result["scores"], dtype=np.float64), atol=1e-6
        )
