# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for ``omni_mercury_engine.ml.mercury_ml`` primitives.

Pins three audit fixes:

* ``IsotonicRegression`` pools tied x values to their mean target before the
  Pool-Adjacent-Violators pass (previously the fit depended on the unstable
  tie order and ``np.interp`` ran over duplicated knots, so four tied inputs
  with targets ``[0, 1, 0, 1]`` predicted 1.0 instead of 0.5).
* ``average_precision_score`` groups samples by unique threshold, making it
  invariant to the input order of tied scores (previously the all-tied case
  scored anywhere between ~0.31 and 1.0 depending on label order).
* ``SGDClassifier`` honors its ``max_iter`` and ``learning_rate`` arguments
  (previously both were stored and ignored).
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.ml.mercury_ml import (
    IsotonicRegression,
    SGDClassifier,
    average_precision_score,
)


class TestIsotonicTiePooling:
    """Tied x values are pooled to their weighted mean before PAV."""

    def test_all_tied_inputs_predict_the_tie_pooled_mean(self) -> None:
        """fit(x=[.5]*4, y=[0,1,0,1]) must predict 0.5 at 0.5, not 1.0."""
        iso = IsotonicRegression().fit(
            np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.0, 1.0, 0.0, 1.0])
        )
        assert iso.predict(np.array([0.5]))[0] == 0.5

    def test_tie_pooling_is_label_order_invariant(self) -> None:
        """Permuting the targets of tied inputs must not change the fit."""
        x = np.array([0.1, 0.1, 0.5, 0.5, 0.9])
        query = np.array([0.1, 0.3, 0.5, 0.9])
        first = IsotonicRegression().fit(x, np.array([0.0, 1.0, 0.0, 1.0, 1.0])).predict(query)
        second = IsotonicRegression().fit(x, np.array([1.0, 0.0, 1.0, 0.0, 1.0])).predict(query)
        np.testing.assert_allclose(first, second)
        # Tie groups pool to their means: 0.5 at x=0.1, 0.5 at x=0.5, 1.0 at 0.9.
        np.testing.assert_allclose(first, [0.5, 0.5, 0.5, 1.0])

    def test_output_is_non_decreasing_on_random_tied_data(self) -> None:
        """Property check: the fitted function is monotone non-decreasing."""
        rng = np.random.RandomState(0)
        for _ in range(25):
            x = rng.rand(80).round(1)  # rounding forces plenty of ties
            y = rng.rand(80)
            model = IsotonicRegression(y_min=0.0, y_max=1.0).fit(x, y)
            queries = np.sort(rng.rand(200))
            predictions = model.predict(queries)
            assert np.all(np.diff(predictions) >= -1e-12)

    def test_untied_strictly_increasing_data_is_interpolated(self) -> None:
        """Already-monotone tie-free data round-trips through the fit."""
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        model = IsotonicRegression().fit(x, y)
        np.testing.assert_allclose(model.predict(x), y)


class TestAveragePrecisionTieAwareness:
    """AP groups tied scores into one threshold, so it is order-invariant."""

    def test_all_tied_scores_give_the_prevalence_for_both_label_orders(self) -> None:
        """100 tied scores at 50% positives must give 0.5 whatever the order."""
        scores = np.full(100, 0.7)
        positives_first = np.array([1] * 50 + [0] * 50)
        negatives_first = np.array([0] * 50 + [1] * 50)
        ap_a = average_precision_score(positives_first, scores)
        ap_b = average_precision_score(negatives_first, scores)
        assert ap_a == ap_b == 0.5

    def test_tie_free_scores_match_the_per_sample_formula(self) -> None:
        """Without ties the grouped AP equals the classic per-sample sum."""
        rng = np.random.RandomState(1)
        for _ in range(20):
            scores = rng.permutation(np.linspace(0.0, 1.0, 64))  # unique scores
            labels = (rng.rand(64) < 0.3).astype(int)
            if labels.sum() == 0:
                labels[0] = 1
            order = np.argsort(scores)[::-1]
            sorted_labels = labels[order]
            tps = np.cumsum(sorted_labels == 1).astype(float)
            precision = tps / np.arange(1, 65)
            recall_change = np.diff(np.concatenate([[0.0], tps / labels.sum()]))
            expected = float(np.sum(precision * recall_change))
            assert abs(average_precision_score(labels, scores) - expected) < 1e-12

    def test_shuffling_tied_samples_does_not_change_ap(self) -> None:
        """Order invariance on data with heavy (but not total) ties."""
        rng = np.random.RandomState(2)
        scores = rng.choice([0.2, 0.5, 0.8], size=120)
        labels = (rng.rand(120) < 0.4).astype(int)
        baseline = average_precision_score(labels, scores)
        for seed in range(5):
            perm = np.random.RandomState(seed).permutation(120)
            assert abs(average_precision_score(labels[perm], scores[perm]) - baseline) < 1e-12


class TestSGDClassifierWiring:
    """max_iter and learning_rate are wired, with defaults preserving behavior."""

    @staticmethod
    def _data() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(3)
        features = rng.randn(30, 4)
        labels = (features[:, 0] > 0).astype(int)
        return features, labels

    def test_max_iter_changes_the_iteration_count(self) -> None:
        """fit() runs max_iter passes; the update counter proves it."""
        features, labels = self._data()
        one_pass = SGDClassifier(max_iter=1)
        one_pass.fit(features, labels)
        five_passes = SGDClassifier(max_iter=5)
        five_passes.fit(features, labels)
        assert one_pass._t == len(features)
        assert five_passes._t == 5 * len(features)

    def test_partial_fit_stays_a_single_pass(self) -> None:
        """partial_fit semantics are untouched by max_iter (back-compat)."""
        features, labels = self._data()
        clf = SGDClassifier(max_iter=5)
        clf.partial_fit(features, labels, classes=np.array([0, 1]))
        assert clf._t == len(features)

    def test_constant_schedule_preserves_the_historical_updates(self) -> None:
        """The default schedule is eta0 at every step, as before the wiring."""
        clf = SGDClassifier(learning_rate="constant", eta0=0.01)
        assert clf._eta(1) == clf._eta(1000) == 0.01

    def test_invscaling_schedule_decays(self) -> None:
        """invscaling must shrink the step size as updates accumulate."""
        clf = SGDClassifier(learning_rate="invscaling", eta0=0.1)
        assert clf._eta(1) == 0.1
        assert clf._eta(100) < clf._eta(1)
        features, labels = self._data()
        clf.fit(features, labels)  # the schedule is exercised end to end
        assert clf._w is not None

    def test_unknown_schedule_fails_loud(self) -> None:
        """A typo'd schedule must raise instead of being silently ignored."""
        import pytest

        clf = SGDClassifier(learning_rate="bogus")
        features, labels = self._data()
        with pytest.raises(ValueError, match="learning_rate"):
            clf.fit(features, labels)
