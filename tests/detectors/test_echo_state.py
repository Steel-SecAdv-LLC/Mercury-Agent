# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the EchoStateDetector regime-novelty detector."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.echo_state import EchoStateDetector


def _series_with_segment(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 800)
    x[400:500] = rng.normal(3.0, 2.5, size=100)  # sustained regime shift
    return x.astype(np.float64)


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = EchoStateDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=400))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x = _series_with_segment()
        scores = np.asarray(EchoStateDetector().fit(x).detect(x)["scores"])
        assert scores.shape == (len(x),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _series_with_segment()
        feats = EchoStateDetector().fit(x).extract_features(x)
        assert feats.shape == (len(x), 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            EchoStateDetector(reservoir_size=1)
        with pytest.raises(ValueError):
            EchoStateDetector(spectral_radius=0.0)

    def test_deterministic(self) -> None:
        x = _series_with_segment(7)
        a = EchoStateDetector().fit(x).detect(x)["scores"]
        b = EchoStateDetector().fit(x).detect(x)["scores"]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSignal:
    def test_segment_flagged(self) -> None:
        train = np.random.default_rng(1).normal(size=800)
        x = _series_with_segment(2)
        scores = np.asarray(EchoStateDetector().fit(train).detect(x)["scores"])
        seg = scores[400:500]
        assert float(seg.max()) > 0.5
        assert (seg > 0.5).mean() > 0.3

    def test_low_false_positive_rate(self) -> None:
        """Tightened from 0.1 to the rate ``calibration_quantile`` actually promises.

        800 training samples against 101 readout parameters overfit only mildly,
        so the in-sample squash scale passed a 0.1 bar even while it was wrong;
        that is why this test did not catch the short-series case below.
        Measured with the cross-fitted scale: 0.0213 against a nominal 0.02.
        """
        train = np.random.default_rng(3).normal(size=800)
        test = np.random.default_rng(4).normal(size=800)
        scores = np.asarray(
            EchoStateDetector(calibration_quantile=0.98).fit(train).detect(test)["scores"]
        )
        assert (scores > 0.5).mean() < 0.03


def _stationary_ar1(n: int = 800, seed: int = 7) -> np.ndarray:
    """A normal regime that does not drift, so the documented FPR must hold."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal()
    return x


class TestOutOfSampleCalibration:
    """The squash scale decides the alarm boundary, so it must be measured honestly."""

    def test_the_documented_false_alarm_rate_holds_on_a_stationary_regime(self) -> None:
        """It was 0.2375 against a documented 0.02 when the scale was in-sample.

        A ``reservoir_size + 1`` = 101-parameter ridge readout fitted to 400
        samples reproduces its own training data far better than anything it
        has not seen, so an in-sample 98th-percentile residual understates the
        normal-regime residual and the detector alarms on ordinary data.  The
        series here is a stationary AR(1): there is no drift to blame, only the
        calibration.
        """
        series = _stationary_ar1()
        detector = EchoStateDetector().fit(series[:400])

        flags = np.asarray(detector.detect(series[400:])["is_anomaly"], dtype=bool)

        assert flags.mean() <= 0.05, f"{flags.mean():.4f} against a nominal 0.02"

    def test_the_scale_exceeds_the_in_sample_quantile(self) -> None:
        """Measured on this series: 3.87 against the 3.23 an in-sample quantile gives."""
        train = np.random.default_rng(3).normal(size=800)
        detector = EchoStateDetector().fit(train)

        normalised = (train - detector._mean) / detector._std
        states = detector._run_states(train)
        augmented = np.hstack([states, np.ones((states.shape[0], 1))])
        in_sample = np.abs(normalised[1:] - augmented[:-1] @ detector._w_out)
        in_sample_scale = float(np.quantile(in_sample, 0.98)) / float(np.log(2.0))

        assert detector._scale > 1.1 * in_sample_scale

    def test_the_readout_still_trains_on_every_sample(self) -> None:
        """Cross-fitting moves the threshold, not the ranking: ``_w_out`` is unchanged."""
        train = np.random.default_rng(3).normal(size=800)
        detector = EchoStateDetector().fit(train)

        normalised = (train - detector._mean) / detector._std
        states = detector._run_states(train)
        augmented = np.hstack([states, np.ones((states.shape[0], 1))])
        x_train, y_train = augmented[:-1], normalised[1:]
        expected = np.linalg.solve(
            x_train.T @ x_train + detector.ridge * np.eye(x_train.shape[1]),
            x_train.T @ y_train,
        )

        assert detector._w_out is not None
        np.testing.assert_allclose(detector._w_out, expected)

    def test_calibration_folds_is_validated(self) -> None:
        with pytest.raises(ValueError, match="calibration_folds"):
            EchoStateDetector(calibration_folds=1)

    def test_a_series_too_short_to_fold_still_fits(self) -> None:
        """No fold can leave more than half of 101 parameters, so it falls back."""
        detector = EchoStateDetector().fit(np.array([1.0, 2.0, 3.0]))

        scores = np.asarray(detector.detect(np.array([1.0, 9.0]))["scores"])
        assert detector.is_fitted()
        assert np.isfinite(scores).all()
        assert ((scores >= 0.0) & (scores <= 1.0)).all()
