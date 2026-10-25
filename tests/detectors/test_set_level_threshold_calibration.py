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
now calibrates.

The nano composite used to rise with length, which blocked the same cure.
That is fixed here -- its molecular term read a byte-transition *count* over
a fixed constant, and its dimensional term contributed a weighted 0.0 it had
never measured -- so the composite now holds one scale. It still keeps its
hand-set cut, because two of its five terms do not carry the direction a cut
would need: ``_scan_at_scale`` falls when a localised burst dominates the
variance spread, cancelling the convergence term, and ``resonance_score`` is
near-constant across every case tried. Calibrating on top of that would stop
the false alarms by trading them for missed bursts. The detector says so
instead, via ``threshold_calibrated``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.detectors.acceleration_dynamics import AccelerationDynamicsDetector
from omni_mercury_engine.detectors.physics_integration import PhysicsIntegratedDetector
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
        """0.60 sits above the measured normal regime of 0.49 to 0.55."""
        series = stationary_ar1(7)

        result = (
            NanoSafeguardDetector(config={"micro_threshold": 0.60})
            .fit(series[:FIT])
            .detect(series[FIT:])
        )

        assert result["threshold_calibrated"] is True
        assert result["is_anomaly"] is False

    def test_the_composite_is_invariant_in_signal_length(self) -> None:
        """The property that makes a window-taken cut transferable at all.

        The composite used to rise monotonically with length -- 0.3757,
        0.3915, 0.4049, 0.4125, 0.4216, 0.4215 at lengths 50 to 400 -- because
        the molecular term counted byte transitions and divided by a fixed
        100. A count scales with the input (one transition per byte, eight
        bytes per float64 sample), so it saturated from about 270 samples on.
        The term now reads a *rate*, and the abstaining dimensional term no
        longer contributes a weighted 0.0, so the composite holds one scale:
        measured 0.4924, 0.5092 and 0.5048 at lengths 50, 200 and 400.
        """
        series = stationary_ar1(7)
        detector = NanoSafeguardDetector().fit(series[:FIT])

        means = []
        for width in (50, 200, 400):
            windows = np.lib.stride_tricks.sliding_window_view(series, width)[::37][:12]
            means.append(
                float(np.mean([detector.detect_micro_anomalies(w).confidence for w in windows]))
            )

        assert max(means) - min(means) < 0.03

    def test_the_molecular_term_reads_a_rate_not_a_count(self) -> None:
        """The count is still reported; the score reads the rate."""
        series = stationary_ar1(7)
        detector = NanoSafeguardDetector().fit(series[:FIT])

        short = detector.detect_micro_anomalies(series[FIT : FIT + 100])
        long = detector.detect_micro_anomalies(series[FIT : FIT + 400])

        # The count scales with the input, which is why it cannot be scored.
        assert long.bit_level_anomalies > 2 * short.bit_level_anomalies
        # The rate does not.
        assert long.bit_anomaly_rate == pytest.approx(short.bit_anomaly_rate, abs=0.02)

    def test_the_dimensional_term_abstains_on_a_one_dimensional_signal(self) -> None:
        """It measured nothing, so it is omitted rather than scored as 0.0.

        A 1-D array reshapes to ``(n, 1)`` and the projection has nothing to
        reduce, so this term can never fire for the detector's declared
        modality. Keeping its weight while summing a hard 0.0 held the
        composite a quarter lower and made the same cut mean different things
        for 1-D and 2-D input.
        """
        series = stationary_ar1(7)
        detector = NanoSafeguardDetector().fit(series[:FIT])

        result = detector.detect_micro_anomalies(series[FIT:])

        assert result.dimensional_measured is False
        assert detector._dimensional_downsampling_detection(series[FIT:]) is None


class TestPhysicsIntegratedEnforcesItsCalibratedLength:
    """A conformal cut over whole-signal scores only covers one signal length.

    ``PhysicsIntegratedDetector`` fuses component scores that each rise with
    the signal's length, so the fused score does too: a 400-sample signal
    scores 1.0000 where 64-sample signals of the same stationary process
    average 0.83. Measured against a cut calibrated at width 64, the share of
    *normal* signals flagged went 0.00 at the fitted width, 0.20 at 1.12x,
    0.70 at 1.25x, 0.90 at 2x and 1.00 at 6.25x. Scoring one length against
    another's threshold is therefore not a calibrated verdict, and this is
    how the detector came to call entirely normal data anomalous.
    """

    WIDTH = 64

    @staticmethod
    def _stack(series: np.ndarray[Any, Any], width: int) -> np.ndarray[Any, Any]:
        """A 2-D stack of ``width``-sample signals, which is this modality."""
        return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(series, width)[::4])

    def _fitted(self) -> PhysicsIntegratedDetector:
        series = stationary_ar1(7)
        return PhysicsIntegratedDetector().fit(self._stack(series[:FIT], self.WIDTH))

    def test_fit_records_the_length_the_cut_was_measured_at(self) -> None:
        detector = self._fitted()

        assert detector._threshold_calibrated is True
        assert detector._calibration_signal_length == self.WIDTH

    def test_a_signal_of_the_calibrated_length_is_scored(self) -> None:
        series = stationary_ar1(7)
        detector = self._fitted()

        result = detector.detect(series[FIT : FIT + self.WIDTH].reshape(1, -1))

        assert result["threshold_calibrated"] is True
        assert result["calibration_signal_length"] == self.WIDTH
        assert result["is_anomaly"] is False

    @pytest.mark.parametrize("length", [72, 128, 400])
    def test_a_signal_the_cut_does_not_cover_is_declined(self, length: int) -> None:
        """Declined, not scored: the engine skips it and counts it declined."""
        series = stationary_ar1(7)
        detector = self._fitted()

        with pytest.raises(InputModalityError, match="does not transfer"):
            detector.detect(series[FIT : FIT + length].reshape(1, -1))

    def test_a_small_difference_is_still_covered(self) -> None:
        """The band is a measured tolerance, not an equality check."""
        series = stationary_ar1(7)
        detector = self._fitted()

        result = detector.detect(series[FIT : FIT + self.WIDTH + 4].reshape(1, -1))

        assert result["threshold_calibrated"] is True

    def test_an_uncalibrated_cut_enforces_no_length(self) -> None:
        """There is no false-alarm bound to protect, and fit() says so.

        A caller who sets the threshold owns the operating point, so the
        detector does not also refuse their input on length.
        """
        series = stationary_ar1(7)
        detector = PhysicsIntegratedDetector({"threshold": 0.95}).fit(
            self._stack(series[:FIT], self.WIDTH)
        )

        result = detector.detect(series[FIT:].reshape(1, -1))

        assert result["threshold_calibrated"] is False
