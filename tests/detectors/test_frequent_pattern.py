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
