# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tier score alignment keeps time, and the streaming scorer does not flatter itself.

Four defects in ``detectors.detection_tier``, each pinned here:

* ``align_point_scores`` stretched a windowed detector's shorter score vector
  across the series by linear interpolation, moving every score to a point
  its window did not end at, and broadcast a series-level scalar to every
  point with nothing to say it was not a per-point score.
* ``TierStreamingScorer`` answered a message with no number in it with
  ``is_anomaly: False`` -- an all-clear for a reading that was never taken.
* It did not recognise ``np.float32`` / ``np.int64`` values as numbers.
* It refitted the detector on a window that already contained the point it
  was about to score.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from omni_mercury_engine.detectors import detection_tier
from omni_mercury_engine.detectors.detection_tier import (
    TierStreamingScorer,
    align_point_scores,
)


class _WindowedDetector:
    """Scores each full window of ``width`` points by its last point's size."""

    name = "windowed_stub"

    def __init__(self, width: int) -> None:
        self.width = width

    def detect(self, data: Any) -> dict[str, Any]:
        x = np.asarray(data, dtype=float)
        ends = np.arange(self.width - 1, x.size)
        return {"scores": np.clip(np.abs(x[ends]) / 10.0, 0.0, 1.0)}


class _StridedDetector:
    """One score per ``stride`` points, with the end index of each window."""

    name = "strided_stub"

    def detect(self, data: Any) -> dict[str, Any]:
        x = np.asarray(data, dtype=float)
        ends = np.arange(9, x.size, 10)
        return {"scores": np.clip(np.abs(x[ends]) / 10.0, 0.0, 1.0), "score_indices": ends}


class _ScalarDetector:
    name = "scalar_stub"

    def detect(self, data: Any) -> dict[str, Any]:
        return {"anomaly_score": 0.4}


class _PerPointDetector:
    name = "per_point_stub"

    def detect(self, data: Any) -> dict[str, Any]:
        return {"scores": np.linspace(0.0, 1.0, np.asarray(data).size)}


def _spike_series(n: int = 100, at: int = 70) -> np.ndarray:
    x = np.zeros(n)
    x[at] = 10.0
    return x


class TestWindowedScoresLandOnTheirWindowEnd:
    def test_a_window_score_is_placed_at_the_point_its_window_ends_at(self) -> None:
        """A 21-point window over 100 points: the spike at 70 scores at 70."""
        scores = align_point_scores(cast("Any", _WindowedDetector(21)), _spike_series())

        assert scores.shape == (100,)
        assert int(np.argmax(scores)) == 70
        assert scores[70] == pytest.approx(1.0)
        assert np.count_nonzero(scores) == 1

    def test_points_before_the_first_full_window_have_no_score(self) -> None:
        x = np.full(100, 5.0)

        scores = align_point_scores(cast("Any", _WindowedDetector(21)), x)

        np.testing.assert_array_equal(scores[:20], 0.0)
        np.testing.assert_allclose(scores[20:], 0.5)

    def test_declared_end_indices_are_used(self) -> None:
        """A strided detector that names its window ends is placed exactly there."""
        scores = align_point_scores(cast("Any", _StridedDetector()), _spike_series(at=69))

        assert int(np.argmax(scores)) == 69
        assert np.count_nonzero(scores) == 1

    def test_the_detailed_result_says_which_points_carry_their_own_score(self) -> None:
        detailed = detection_tier.align_point_scores_detailed(
            cast("Any", _WindowedDetector(21)), _spike_series()
        )

        assert detailed.alignment == "window_end"
        np.testing.assert_array_equal(detailed.localized[:20], False)
        np.testing.assert_array_equal(detailed.localized[20:], True)
        np.testing.assert_array_equal(
            detailed.scores, align_point_scores(cast("Any", _WindowedDetector(21)), _spike_series())
        )


class TestSeriesLevelScoresAreMarkedCoarse:
    def test_a_scalar_is_reported_as_series_level(self) -> None:
        detailed = detection_tier.align_point_scores_detailed(
            cast("Any", _ScalarDetector()), np.arange(30.0)
        )

        assert detailed.alignment == "series_level"
        assert not detailed.localized.any()
        np.testing.assert_allclose(detailed.scores, 0.4)

    def test_a_per_point_vector_is_reported_as_per_point(self) -> None:
        detailed = detection_tier.align_point_scores_detailed(
            cast("Any", _PerPointDetector()), np.arange(30.0)
        )

        assert detailed.alignment == "per_point"
        assert detailed.localized.all()

    def test_series_level_tier_members_are_marked(self) -> None:
        """``rca`` gives one score per series; the mark says so."""
        from omni_mercury_engine.detectors.detection_tier import build_tier_detectors

        rng = np.random.default_rng(0)
        x = rng.normal(size=120)
        detector = build_tier_detectors(["rca"])["rca"]
        detector.fit(x)

        detailed = detection_tier.align_point_scores_detailed(detector, x)

        assert detailed.alignment == "series_level"
        assert np.ptp(detailed.scores) == 0.0


class _RecordingDetector:
    """Flags a point above everything it was fitted on; remembers its fits."""

    name = "recording_stub"

    def __init__(self) -> None:
        self.fits: list[np.ndarray] = []
        self._ceiling: float | None = None

    def is_fitted(self) -> bool:
        return self._ceiling is not None

    def fit(self, data: Any) -> _RecordingDetector:
        x = np.asarray(data, dtype=float).copy()
        self.fits.append(x)
        self._ceiling = float(np.max(x))
        return self

    def detect(self, data: Any) -> dict[str, Any]:
        x = np.asarray(data, dtype=float)
        assert self._ceiling is not None
        return {"scores": (x > self._ceiling).astype(float)}


def _scorer(detector: Any, **kwargs: Any) -> TierStreamingScorer:
    return TierStreamingScorer(cast("Any", detector), name="stub", **kwargs)


class TestAMessageWithoutANumberIsAnAbstention:
    def _ready(self) -> TierStreamingScorer:
        scorer = _scorer(_RecordingDetector(), window_size=16, min_samples=4)
        for value in (1.0, 2.0, 1.5, 1.2, 1.1):
            scorer({"value": value})
        return scorer

    @pytest.mark.parametrize(
        "message",
        [{"tag": "note"}, {"value": float("nan")}, {"value": float("inf")}, {"value": None}, {}],
    )
    def test_no_verdict_is_given(self, message: dict[str, Any]) -> None:
        result = self._ready()(message)

        assert result["is_anomaly"] is None
        assert result["abstain"] is True
        assert result["reason"] == "no_finite_numeric_value"
        assert result["warmup"] is False
        # The pipeline's score statistics need a number; it is a placeholder.
        assert result["score"] == 0.0

    def test_a_scored_message_does_not_abstain(self) -> None:
        result = self._ready()({"value": 1.3})

        assert result["is_anomaly"] is False
        assert result["abstain"] is False

    def test_an_abstention_leaves_the_window_untouched(self) -> None:
        scorer = self._ready()
        before = list(scorer._buffer)

        scorer({"tag": "note"})

        assert list(scorer._buffer) == before


class TestNumpyScalarsAreNumbers:
    @pytest.mark.parametrize(
        "value",
        [np.float32(2.5), np.float64(2.5), np.int64(3), np.int32(3), np.uint8(3), Fraction(5, 2)],
    )
    def test_a_numpy_scalar_is_extracted(self, value: Any) -> None:
        scorer = _scorer(_RecordingDetector())

        assert scorer._extract_value({"value": value}) == pytest.approx(float(value))

    @pytest.mark.parametrize("value", [True, np.bool_(True), "3.0", None, 1 + 2j, np.complex64(1)])
    def test_a_non_real_value_is_not(self, value: Any) -> None:
        scorer = _scorer(_RecordingDetector())

        assert scorer._extract_value({"value": value}) is None

    def test_a_float32_stream_is_scored(self) -> None:
        """Every reading used to be dropped: the scorer never left warm-up."""
        scorer = _scorer(_RecordingDetector(), window_size=16, min_samples=4)

        results = [scorer({"value": np.float32(v)}) for v in (1.0, 2.0, 1.5, 1.2, 1.1, 9.0)]

        assert results[-1]["warmup"] is False
        assert results[-1]["is_anomaly"] is True


class TestTheNewestPointIsNotInItsOwnFit:
    def test_the_fit_window_ends_before_the_point_being_scored(self) -> None:
        detector = _RecordingDetector()
        scorer = _scorer(detector, window_size=8, min_samples=4, refit_interval=1)

        for value in (1.0, 2.0, 1.5, 1.2, 50.0):
            result = scorer({"value": value})

        np.testing.assert_array_equal(detector.fits[-1], [1.0, 2.0, 1.5, 1.2])
        assert 50.0 not in detector.fits[-1]
        # Fitted on the past only, the detector sees the spike as new.
        assert result["anomaly_score"] == 1.0
        assert result["is_anomaly"] is True

    def test_a_spike_on_a_refit_tick_is_not_absorbed_into_the_baseline(self) -> None:
        """Refitting every tick: a spike was always inside its own baseline."""
        detector = _RecordingDetector()
        scorer = _scorer(detector, window_size=32, min_samples=8, refit_interval=1)
        rng = np.random.default_rng(4)
        values = list(rng.uniform(0.0, 1.0, size=40))
        values[30] = 25.0

        flags = [scorer({"value": v})["is_anomaly"] for v in values]

        assert flags[30] is True

    def test_scoring_starts_once_min_samples_are_buffered(self) -> None:
        """``min_samples`` points are buffered first; the next one is scored."""
        scorer = _scorer(_RecordingDetector(), window_size=16, min_samples=4)

        results = [scorer({"value": float(v)}) for v in range(6)]

        assert [r["warmup"] for r in results] == [True, True, True, True, False, False]

    def test_a_full_window_is_fitted_on_the_points_before_the_newest(self) -> None:
        detector = _RecordingDetector()
        scorer = _scorer(detector, window_size=4, min_samples=2, refit_interval=1)

        for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            scorer({"value": value})

        np.testing.assert_array_equal(detector.fits[-1], [2.0, 3.0, 4.0, 5.0])
