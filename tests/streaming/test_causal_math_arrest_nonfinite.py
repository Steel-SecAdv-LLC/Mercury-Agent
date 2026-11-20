# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A non-finite reading is a failed sensor reading, not a sample.

``CausalMathArrest.update`` appended whatever it was given to the trailing
window. One NaN therefore sat in the next ``window`` windows, each of which
scored NaN; ``NaN > threshold`` is False, so every one of those ticks was
reported as ``is_anomaly: False``; and the trajectory monitor took
``max(nan - threshold, 0.0)``, which is NaN, for the onset of an excess.

The reading is now refused at the door: it does not enter the history, the
monitor is left as it was, and the tick is reported as an abstention.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omni_mercury_engine.streaming.causal_math_arrest import CausalMathArrest

WINDOW = 32
BAD_AT = 20


def _warmup_and_stream() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    series = np.sin(np.linspace(0.0, 30.0, 300)) + 0.05 * rng.standard_normal(300)
    return series[:200], series[200:]


@pytest.fixture(scope="module")
def clean_scores() -> np.ndarray:
    """Scores of the stream with no bad reading in it."""
    warmup, stream = _warmup_and_stream()
    scores, _ = CausalMathArrest(window=WINDOW).fit(warmup).score_stream(stream)
    return scores


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
class TestABadReadingDoesNotEnterTheHistory:
    def test_later_scores_are_those_of_the_stream_without_it(
        self, bad: float, clean_scores: np.ndarray
    ) -> None:
        """Every later tick scores exactly as if the bad tick had not happened."""
        warmup, stream = _warmup_and_stream()
        with_bad = np.insert(stream, BAD_AT, bad)

        details = CausalMathArrest(window=WINDOW).fit(warmup).score_stream_detailed(with_bad)

        scores = np.asarray(details["scores"])
        assert math.isnan(scores[BAD_AT])
        np.testing.assert_array_equal(np.delete(scores, BAD_AT), clean_scores)
        assert np.isfinite(np.delete(scores, BAD_AT)).all()

    def test_the_tick_is_reported_as_an_abstention(self, bad: float) -> None:
        warmup, stream = _warmup_and_stream()
        detector = CausalMathArrest(window=WINDOW).fit(warmup)
        for value in stream[:BAD_AT]:
            detector.update(float(value))

        result = detector.update(bad)

        assert result["is_anomaly"] is None
        assert result["abstain"] is True
        assert result["reason"] == "non_finite_input"
        assert math.isnan(result["score"])
        assert result["trajectory_stable"] is None
        assert result["window_used"] == WINDOW

    def test_the_stream_arrays_mark_it(self, bad: float) -> None:
        warmup, stream = _warmup_and_stream()
        with_bad = np.insert(stream[:40], BAD_AT, bad)

        details = CausalMathArrest(window=WINDOW).fit(warmup).score_stream_detailed(with_bad)

        assert bool(details["abstain"][BAD_AT]) is True
        assert bool(details["alarms"][BAD_AT]) is False
        assert not np.delete(np.asarray(details["abstain"]), BAD_AT).any()

    def test_the_history_is_untouched(self, bad: float) -> None:
        warmup, stream = _warmup_and_stream()
        detector = CausalMathArrest(window=WINDOW).fit(warmup)
        for value in stream[:BAD_AT]:
            detector.update(float(value))
        before = list(detector._history)

        detector.update(bad)

        assert detector._history == before
        assert np.isfinite(detector._history).all()


class TestTheTrajectoryMonitorIsLeftAlone:
    def test_a_bad_reading_is_not_an_excess_onset(self) -> None:
        """``max(nan - threshold, 0.0)`` is NaN, which read as an onset."""
        warmup, stream = _warmup_and_stream()
        detector = CausalMathArrest(window=WINDOW).fit(warmup)
        detector.update(float(stream[0]))
        assert detector._excess_onset is None

        detector.update(float("nan"))

        assert detector._excess_onset is None
        assert detector._steps_since_onset == 0
        assert detector._violation_streak == 0

    def test_an_excess_in_progress_is_neither_advanced_nor_reset(self) -> None:
        warmup, _stream = _warmup_and_stream()
        detector = CausalMathArrest(window=WINDOW).fit(warmup)
        threshold = detector.threshold
        assert threshold is not None
        for _ in range(5):
            detector._trajectory_check(threshold + 0.3, threshold)
        state = (detector._excess_onset, detector._steps_since_onset, detector._violation_streak)

        detector.update(float("nan"))

        assert (
            detector._excess_onset,
            detector._steps_since_onset,
            detector._violation_streak,
        ) == state


class TestOrdinaryTicksAreUnchanged:
    def test_a_finite_tick_gives_a_verdict_and_no_reason(self) -> None:
        warmup, stream = _warmup_and_stream()
        detector = CausalMathArrest(window=WINDOW).fit(warmup)

        result = detector.update(float(stream[0]))

        assert isinstance(result["is_anomaly"], bool)
        assert result["reason"] is None
        assert result["abstain"] == (not result["trajectory_stable"])

    def test_a_bad_reading_in_the_warmup_does_not_seed_the_history(self) -> None:
        warmup, stream = _warmup_and_stream()
        dirty = warmup.copy()
        dirty[-5] = float("nan")

        detector = CausalMathArrest(window=WINDOW).fit(dirty)

        assert np.isfinite(detector._history).all()
        assert len(detector._history) == WINDOW
        assert math.isfinite(detector.update(float(stream[0]))["score"])
