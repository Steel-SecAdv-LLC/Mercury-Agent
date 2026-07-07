# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the frequent-pattern / association-rule violation detector."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.frequent_pattern import FrequentPatternDetector


def _transactions(seed: int = 0, n: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.zeros((n, 6), dtype=np.int64)
    for i in range(n):
        if rng.random() < 0.7:
            t[i, 0] = t[i, 1] = t[i, 2] = 1  # rule A,B => C
        t[i, 3:] = (rng.random(3) < 0.3).astype(np.int64)
    return t


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = FrequentPatternDetector()
        assert det.is_fitted() is False
        det.fit(_transactions(0, 200))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        t = _transactions()
        scores = np.asarray(FrequentPatternDetector().fit(t).detect(t)["scores"])
        assert scores.shape == (t.shape[0],)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        t = _transactions()
        feats = FrequentPatternDetector().fit(t).extract_features(t)
        assert feats.shape == (t.shape[0], 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            FrequentPatternDetector(min_support=0.0)
        with pytest.raises(ValueError):
            FrequentPatternDetector(min_confidence=1.5)

    def test_deterministic(self) -> None:
        t = _transactions(7)
        a = FrequentPatternDetector().fit(t).detect(t)["scores"]
        b = FrequentPatternDetector().fit(t).detect(t)["scores"]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSignal:
    def test_rule_violation_flagged(self) -> None:
        train = _transactions(1)
        test = train.copy()
        test[50, 0] = test[50, 1] = 1
        test[50, 2] = 0  # A,B present but C absent -> violates A,B => C
        scores = np.asarray(FrequentPatternDetector().fit(train).detect(test)["scores"])
        assert float(scores[50]) > 0.5

    def test_low_false_positive_rate(self) -> None:
        train = _transactions(3)
        test = _transactions(4)
        scores = np.asarray(
            FrequentPatternDetector(calibration_quantile=0.98).fit(train).detect(test)["scores"]
        )
        assert (scores > 0.5).mean() < 0.1


class TestRobustness:
    """Mining must stay bounded on wide / degenerate input (no combinatorial blow-up)."""

    def test_single_wide_transaction_is_bounded(self) -> None:
        # A 1-D continuous series reshapes to a single 256-item transaction where
        # every present item has support 1.0 and nothing prunes -- the pathological
        # case that must not explode combinatorially in the fusion registry.
        rng = np.random.default_rng(0)
        x = rng.normal(size=256)
        det = FrequentPatternDetector().fit(x)
        result = det.detect(x)
        score = float(result["anomaly_score"])
        assert 0.0 <= score <= 1.0

    def test_max_items_caps_frequent_itemsets(self) -> None:
        # Every column present in every transaction => all singletons frequent;
        # the per-level cap keeps only max_items of them.
        mat = np.ones((32, 40), dtype=np.int64)
        det = FrequentPatternDetector(min_support=0.1, max_items=8).fit(mat)
        support = det._mine_frequent(det._to_bool_matrix(mat))
        singletons = [iset for iset in support if len(iset) == 1]
        assert len(singletons) <= 8

    def test_invalid_max_items(self) -> None:
        with pytest.raises(ValueError):
            FrequentPatternDetector(max_items=1)

    def test_wide_continuous_matrix_terminates(self) -> None:
        rng = np.random.default_rng(2)
        big = rng.normal(size=(64, 200))
        det = FrequentPatternDetector().fit(big)
        scores = np.asarray(det.detect(big)["scores"])
        assert scores.shape == (64,)
        assert np.isfinite(scores).all()

    def test_scalar_input_no_crash(self) -> None:
        # Regression: 0-D (scalar) input was not normalised to 2-D and crashed the
        # downstream indexing; it must degrade to a single 1-item transaction.
        det = FrequentPatternDetector()
        assert det._to_bool_matrix(np.array(5.0)).shape == (1, 1)
        det.fit(np.array(3.0))
        scores = np.asarray(det.detect(np.array(3.0))["scores"], dtype=float)
        assert np.all(np.isfinite(scores))
