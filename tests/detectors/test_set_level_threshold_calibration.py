# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Set-level detectors must not decide on a cut their own scores sit above.

A set-level detector reduces a whole signal to one score. Several shipped a
hand-set or default cut that the *normal* regime of that score exceeds, so
they called an entirely normal stationary series anomalous every time:

* ``AccelerationDynamicsDetector`` decided at the 0.5 ``BaseDetector``
  default while its fused score measures 0.79 to 0.83 on stationary input.
* ``NanoSafeguardDetector`` decides at a hand-set 0.05 while its composite
  measures 0.40 to 0.44 on the same input.

Whether that can be calibrated from the training signal turns on one
measured property: does the score depend on the signal's length? Windows of
a training signal are the only source of repeated scores when the caller
supplies one series, so a cut taken on them is only valid if it transfers.

It does for the kinematic score, which is flat in length, so that detector
now calibrates. It does not for the nano composite, which rises with length,
so that one keeps its hand-set cut and says so rather than implying a bound
it does not have.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.detectors.acceleration_dynamics import AccelerationDynamicsDetector
from omni_mercury_engine.safeguards.nano_safeguards import NanoSafeguardDetector

SEEDS = (7, 11, 23, 101, 999)
FIT = 400


def stationary_ar1(seed: int, n: int = 800, phi: float = 0.6) -> np.ndarray[Any, Any]:
    """A normal regime with no level drift: ``x[t] = phi * x[t-1] + e[t]``."""
    rng = np.random.default_rng(seed)
    series = np.zeros(n)
    for index in range(1, n):
        series[index] = phi * series[index - 1] + rng.normal()
    return series


class TestAccelerationDynamicsCalibratesItsCut:
    """The kinematic score is flat in length, so windows calibrate it."""

    def test_a_normal_series_is_not_called_anomalous(self) -> None:
        """Every one of the five seeds used to come back anomalous."""
        verdicts = []
        for seed in SEEDS:
            series = stationary_ar1(seed)
            detector = AccelerationDynamicsDetector().fit(series[:FIT])
            verdicts.append(bool(detector.detect(series[FIT:])["is_anomaly"]))

        assert not any(verdicts)

    def test_fit_reports_the_cut_as_calibrated(self) -> None:
        series = stationary_ar1(7)

        result = AccelerationDynamicsDetector().fit(series[:FIT]).detect(series[FIT:])

        assert result["threshold_calibrated"] is True
        assert result["threshold"] != 0.5

    @pytest.mark.parametrize(
        ("label", "injure"),
        [
            ("level shift", lambda x: x + 6.0),
            ("volatility", lambda x: x * 5.0),
        ],
    )
    def test_real_anomalies_still_fire(self, label: str, injure: Any) -> None:
        """Calibrating the cut must not amount to switching the alarm off.

        ``label`` names the case in the test id.
        """
        series = stationary_ar1(7)
        detector = AccelerationDynamicsDetector().fit(series[:FIT])
        held_out = series[FIT:].copy()
        injured = np.concatenate([held_out[:150], injure(held_out[150:200]), held_out[200:]])

        assert detector.detect(injured)["is_anomaly"]

    def test_an_explicit_threshold_survives_fit(self) -> None:
        series = stationary_ar1(7)

        detector = AccelerationDynamicsDetector(config={"threshold": 0.5}).fit(series[:FIT])

        assert detector.threshold == 0.5
        assert detector._threshold_calibrated is False

    def test_the_score_is_flat_in_signal_length(self) -> None:
        """The property the window calibration rests on.

        If this stops holding, a cut taken on windows no longer transfers to
        a whole signal and the calibration above is no longer sound.
        """
        series = stationary_ar1(7)
        detector = AccelerationDynamicsDetector().fit(series[:FIT])

        means = []
        for width in (100, 200, 400):
            windows = np.lib.stride_tricks.sliding_window_view(series, width)[::37][:12]
            means.append(float(np.mean([detector.detect(w)["anomaly_score"] for w in windows])))

        assert max(means) - min(means) < 0.05


class TestNanoSafeguardDeclaresItsCutUncalibrated:
    """The nano composite rises with length, so windows cannot calibrate it."""

    def test_an_uncalibrated_verdict_says_so(self) -> None:
        """The verdict is still reached, but it claims no false-alarm bound."""
        series = stationary_ar1(7)

        result = NanoSafeguardDetector().fit(series[:FIT]).detect(series[FIT:])

        assert result["threshold_calibrated"] is False

    def test_a_caller_supplied_cut_is_reported_as_calibrated(self) -> None:
        series = stationary_ar1(7)

        result = (
            NanoSafeguardDetector(config={"micro_threshold": 0.45})
            .fit(series[:FIT])
            .detect(series[FIT:])
        )

        assert result["threshold_calibrated"] is True
        assert result["is_anomaly"] is False

    def test_the_score_rises_with_signal_length(self) -> None:
        """Why windows are not a sound calibration set for this detector.

        Measured 0.3757 to 0.4215 across lengths 50 to 400 on one process. A
        cut taken on shorter windows therefore sits below where a longer
        signal scores, which is what left two of the five seeds flagged when
        window calibration was tried here.
        """
        series = stationary_ar1(7)
        detector = NanoSafeguardDetector().fit(series[:FIT])

        means = []
        for width in (50, 200, 400):
            windows = np.lib.stride_tricks.sliding_window_view(series, width)[::37][:12]
            means.append(
                float(np.mean([detector.detect_micro_anomalies(w).confidence for w in windows]))
            )

        assert means == sorted(means)
        assert means[-1] - means[0] > 0.02
