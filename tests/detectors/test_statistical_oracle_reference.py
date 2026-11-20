# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Oracle reads columns as signals and never moves the decision scores.

Four fit-time truths of :class:`MercuryAnomalyDetector`, each of which the
prior code broke:

* the spectral Oracle analysed ROWS (one time step across features) as
  "signals" at an assumed 1 kHz, so its references described d-sample
  fragments rather than any feature's series;
* its batch-scalar influence multiplier was multiplied into the decision
  scores before the ``[0, 1]`` clip, shifting the operating point per batch
  and saturating top-ranked scores into ties on anomalous-looking batches;
* ``fit_with_labels`` stored the conformal reference under the UNSUPERVISED
  weights and then replaced the weights, so p-values and FDR flags were
  computed against a stale score function (measured up to 0.287 apart);
* the reference included labelled anomalies, inflating the conformal cut.

Measuring ensemble diversity also halved component weights as a side effect;
the penalty is now an explicit fit step.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def _temporal_data(n: int = 400, d: int = 2, seed: int = 7) -> np.ndarray:
    """Smooth multivariate series that classifies as TEMPORAL."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    cols = [
        np.sin(2 * np.pi * t / 37.0 + k)
        + 0.1 * rng.normal(size=n)
        + np.cumsum(rng.normal(0, 0.05, n))
        for k in range(d)
    ]
    return np.column_stack(cols)


@pytest.fixture(scope="module")
def fitted() -> MercuryAnomalyDetector:
    det = MercuryAnomalyDetector()
    det.fit(_temporal_data())
    return det


class TestOracleAxis:
    def test_oracle_is_fitted_on_columns_not_rows(self, fitted: MercuryAnomalyDetector) -> None:
        """Each per-band reference array holds one entry per FEATURE signal."""
        oracle = fitted._oracle_detector
        assert oracle is not None, "temporal fit should arm the Oracle"
        n_signals = {len(v) for v in oracle._ref_band_powers.values()}
        assert n_signals == {2}, (
            f"per-band references describe {n_signals} signals; the fit handed "
            "the Oracle rows (400 d-sample fragments) instead of the 2 columns"
        )

    def test_oracle_inactive_below_min_rows(self, fitted: MercuryAnomalyDetector) -> None:
        """A window too short for band statistics reports an inactive Oracle."""
        fitted.detect(_temporal_data(n=16, seed=11))
        assert fitted._oracle_metadata["active"] is False

    def test_min_rows_guard_also_applies_at_fit(self) -> None:
        det = MercuryAnomalyDetector()
        det.fit(_temporal_data(n=24, seed=3))
        assert det._oracle_detector is None


class TestOracleIsMetadataOnly:
    def test_scores_identical_with_and_without_oracle(self) -> None:
        """The influence multiplier is severity context, not a score factor."""
        batch = _temporal_data(n=256, seed=21)
        det = MercuryAnomalyDetector()
        det.fit(_temporal_data())
        with_oracle = np.asarray(det.detect(batch)["scores"], dtype=np.float64)
        assert det._oracle_metadata["active"] is True
        assert det._oracle_metadata["mean_multiplier"] > 0.0

        det._oracle_detector = None
        without = np.asarray(det.detect(batch)["scores"], dtype=np.float64)
        np.testing.assert_array_equal(with_oracle, without)


class TestReferenceAnchoring:
    @staticmethod
    def _labelled(seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        normal = rng.normal(0.0, 1.0, size=(220, 4))
        anomalies = rng.normal(7.0, 1.0, size=(30, 4))
        data = np.vstack([normal, anomalies])
        labels = np.concatenate([np.zeros(220, dtype=int), np.ones(30, dtype=int)])
        return data, labels

    def test_fit_with_labels_reanchors_reference_under_served_weights(self) -> None:
        data, labels = self._labelled()
        det = MercuryAnomalyDetector()
        det.fit_with_labels(data, labels)
        assert det._reference_scores is not None
        stored = det._reference_scores.copy()
        det._fit_reference_distribution(data[labels == 0])
        np.testing.assert_allclose(det._reference_scores, stored, atol=1e-12)

    def test_calibration_labels_exclude_anomalies_from_reference(self) -> None:
        data, labels = self._labelled(seed=9)
        with_labels = MercuryAnomalyDetector()
        with_labels.fit(data, calibration_labels=labels)
        blind = MercuryAnomalyDetector()
        blind.fit(data)
        assert with_labels._reference_scores is not None
        assert blind._reference_scores is not None
        # The labelled anomalies score near the top; keeping them in the
        # reference drags its upper tail up. Excluding them must lower it.
        assert with_labels._reference_scores[-1] <= blind._reference_scores[-1]
        assert len(with_labels._reference_scores) < len(blind._reference_scores)


class TestDiversityPurity:
    def test_measuring_diversity_does_not_move_weights(self) -> None:
        det = MercuryAnomalyDetector()
        arr = _temporal_data(n=200, seed=13)
        det.fit(arr)
        det._component_aucs = {"resonance": 0.9, "kinematic": 0.6, "infogeo": 0.7}
        same = np.linspace(0.0, 1.0, 200)
        det._compute_resonance_score = lambda X: same  # type: ignore[method-assign]
        det._compute_kinematic_score = lambda X: same  # type: ignore[method-assign]
        det._compute_info_geometry_score = lambda X: same  # type: ignore[method-assign]
        before = det._adaptive_weights.copy()
        diversity = det._compute_ensemble_diversity(arr)
        np.testing.assert_array_equal(det._adaptive_weights, before)
        assert diversity["mean_correlation"] > 0.99

    def test_penalty_is_an_explicit_step(self) -> None:
        det = MercuryAnomalyDetector()
        det.fit(_temporal_data(n=200, seed=17))
        det._component_aucs = {"resonance": 0.9, "kinematic": 0.6, "infogeo": 0.7}
        det._adaptive_weights = np.array([0.4, 0.3, 0.3])
        det._apply_diversity_penalty(
            {
                "resonance_kinematic": 0.95,
                "resonance_infogeo": 0.0,
                "kinematic_infogeo": 0.0,
                "mean_correlation": 0.4,
            }
        )
        # kinematic (lower AUC in the correlated pair) halved, then renormalized.
        expected = np.array([0.4, 0.15, 0.3])
        expected = expected / expected.sum()
        np.testing.assert_allclose(det._adaptive_weights, expected, atol=1e-12)
