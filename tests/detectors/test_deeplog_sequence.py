# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DeepLog-style sequence/log-template detector."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.deeplog_sequence import DeepLogSequenceDetector


def _grammar(n: int = 300) -> np.ndarray:
    return np.tile(np.array([0, 1, 2, 3, 4]), n)


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = DeepLogSequenceDetector()
        assert det.is_fitted() is False
        det.fit(_grammar(50))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        seq = _grammar(50)
        scores = np.asarray(DeepLogSequenceDetector().fit(seq).detect(seq)["scores"])
        assert scores.shape == (len(seq),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        seq = _grammar(50)
        feats = DeepLogSequenceDetector().fit(seq).extract_features(seq)
        assert feats.shape == (len(seq), 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            DeepLogSequenceDetector(order=0)
        with pytest.raises(ValueError):
            DeepLogSequenceDetector(top_g=0)

    def test_deterministic(self) -> None:
        seq = _grammar(50)
        a = DeepLogSequenceDetector().fit(seq).detect(seq)["scores"]
        b = DeepLogSequenceDetector().fit(seq).detect(seq)["scores"]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSignal:
    def test_out_of_grammar_flagged(self) -> None:
        train = _grammar(300)
        test = train.copy()
        test[700:710] = 9  # keys never seen in training
        out = DeepLogSequenceDetector().fit(train).detect(test)
        scores = np.asarray(out["scores"])
        assert float(scores[700:715].max()) > 0.5
        assert out["metadata"]["top_g_violations"] >= 1

    def test_low_false_positive_rate(self) -> None:
        train = _grammar(300)
        scores = np.asarray(DeepLogSequenceDetector().fit(train).detect(train)["scores"])
        assert (scores > 0.5).mean() < 0.05


class TestDegenerateInput:
    def test_unfitted_detect_is_neutral_not_saturated(self) -> None:
        # Regression: unfitted detect returned probs=0 -> 1 - 0 = 1, flagging every
        # position maximally anomalous. It must degrade to neutral instead.
        det = DeepLogSequenceDetector()
        scores = np.asarray(det.detect(np.array([1, 2, 3, 4, 5]))["scores"], dtype=float)
        assert float(scores.max()) <= 0.5

    def test_to_1d_int_overflow_safe(self) -> None:
        # Regression: a bare np.nan_to_num turned +inf into 1.7977e308, which then
        # overflowed the int64 cast to an implementation-defined garbage key.
        det = DeepLogSequenceDetector()
        keys = det._to_1d_int(np.array([1.0, np.inf, -np.inf, 1e120, np.nan, -7.0]))
        assert keys.dtype == np.int64
        assert np.iinfo(np.int64).min not in keys.tolist()  # no overflow wraparound
        assert np.all(np.abs(keys) <= 2**53)
