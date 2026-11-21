# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the streaming ingest boundary: one bad sample must not end a stream.

Three surfaces take one observation at a time from a source the process does
not control, for as long as the process lives:

* ``StreamingDetector.ingest`` / ``ingest_batch`` (rolling-window detector);
* ``EngineStreamSession.observe`` (the engine's per-sample stream);
* ``HysteresisGate.update`` (the two-threshold verdict gate).

Hypothesis drives arbitrary observations through them -- scalars and vectors
of any width, NaN, infinities, empty samples -- and asserts what must hold
for *any* stream.

The fuzzer found a real defect on its first run, pinned here.  An
observation whose width contradicted the stream's was appended to the window
anyway; the next refit stacked the window into one array and raised
``ValueError`` from numpy, and it raised *before* that tick's sample was
appended -- so the window never advanced and **every later** ``ingest``
raised the same error.  One malformed sample ended the stream until
``reset()``.  The contradiction is now refused at the sample that carries it
(``ValueError``, ``last_status == "rejected"``, counted) and the stream goes
on; ``TestOneBadSampleDoesNotEndTheStream`` fails on the prior code.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import (
    HealthCheck,
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.streaming.engine_stream import EngineStreamSession
from omni_mercury_engine.streaming.hysteresis import HysteresisGate, hysteresis_verdicts
from omni_mercury_engine.streaming.streaming_detector import StreamingDetector

_floats = st.floats(allow_nan=True, allow_infinity=True, width=64)
_finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
# One observation: a scalar, or a vector of 0 to 4 values (0 = an empty sample).
_point = st.one_of(_floats, st.lists(_floats, min_size=0, max_size=4))
_SLOW = [HealthCheck.too_slow, HealthCheck.data_too_large]


def _shape_of(point: Any) -> tuple[int, ...]:
    """The shape ``ingest`` gives one observation (a scalar is one feature)."""
    return (1,) if isinstance(point, float) else (len(point),)


def _detector() -> StreamingDetector:
    return StreamingDetector(window_size=32, refit_interval=5, min_samples=4, score_window=8)


class TestStreamingDetectorIngest:
    @settings(max_examples=120, deadline=None, suppress_health_check=_SLOW)
    @given(points=st.lists(_point, min_size=1, max_size=30))
    def test_only_a_shape_contradiction_raises_and_it_is_counted(self, points: list[Any]) -> None:
        detector = _detector()
        stream_shape: tuple[int, ...] | None = None
        accepted = rejected = 0
        for point in points:
            shape = _shape_of(point)
            contradicts = 0 in shape or (stream_shape is not None and shape != stream_shape)
            if contradicts:
                with pytest.raises(ValueError, match="observation"):
                    detector.ingest(point)
                rejected += 1
                assert detector.last_status == "rejected"
            else:
                result = detector.ingest(point)  # NaN / inf included: must not raise
                assert result is None or isinstance(result, dict)
                stream_shape = shape
                accepted += 1
                assert detector.last_status != "rejected"
        stats = detector.get_stats()
        assert stats["total_points_ingested"] == accepted
        assert stats["rejected_points"] == rejected
        # A refused observation was never kept: the window stacks cleanly.
        assert detector.window_data.shape[0] == min(accepted, detector.window_size)

    @settings(max_examples=60, deadline=None, suppress_health_check=_SLOW)
    @given(
        width=st.integers(min_value=1, max_value=3),
        batches=st.lists(
            st.tuples(st.integers(min_value=0, max_value=4), st.integers(min_value=1, max_value=5)),
            min_size=1,
            max_size=8,
        ),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    def test_a_batch_of_another_width_is_refused_whole(
        self, width: int, batches: list[tuple[int, int]], seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        detector = _detector()
        detector.ingest_batch(rng.normal(size=(6, width)))
        kept = 6
        for batch_width, rows in batches:
            batch = rng.normal(size=(rows, batch_width))
            if batch_width != width:
                with pytest.raises(ValueError, match="observation"):
                    detector.ingest_batch(batch)
            else:
                detector.ingest_batch(batch)
                kept += rows
            assert detector.get_stats()["total_points_ingested"] == kept
        assert detector.window_data.shape == (min(kept, detector.window_size), width)


class TestOneBadSampleDoesNotEndTheStream:
    """Fails on the prior code, where the stream raised on every later tick."""

    def _feed_healthy(self, detector: StreamingDetector, count: int) -> int:
        rng = np.random.default_rng(3)
        return sum(detector.ingest(float(value)) is not None for value in rng.normal(size=count))

    @pytest.mark.parametrize("bad", [[1.0, 2.0], [], [1.0, 2.0, 3.0]])
    def test_the_stream_scores_again_after_a_contradicting_sample(self, bad: list[float]) -> None:
        detector = _detector()
        self._feed_healthy(detector, 6)
        with pytest.raises(ValueError, match="observation"):
            detector.ingest(bad)
        verdicts = self._feed_healthy(detector, 20)  # the prior code raised here, every tick
        assert verdicts == 20
        assert detector.last_status == "ok"
        assert detector.get_stats()["rejected_points"] == 1

    def test_a_contradicting_batch_does_not_end_the_stream(self) -> None:
        detector = _detector()
        self._feed_healthy(detector, 6)
        with pytest.raises(ValueError, match="observation"):
            detector.ingest_batch([[1.0, 2.0], [3.0, 4.0]])
        assert self._feed_healthy(detector, 20) == 20
        assert detector.get_stats()["rejected_points"] == 2

    def test_reset_forgets_the_stream_shape(self) -> None:
        detector = _detector()
        detector.ingest(1.0)
        detector.reset()
        assert detector.ingest([1.0, 2.0]) is None  # a new stream may have a new width
        assert detector.get_stats()["rejected_points"] == 0


class _StepDetector:
    """Deterministic point detector: a sample above 1.0 is an anomaly."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def fit(self, window: Any) -> None:
        """Stateless reference; nothing to fit."""

    def get_effective_threshold(self) -> float:
        return 0.5


class _StepEngine:
    """The minimal engine surface a stream session consumes, deterministic."""

    def __init__(self) -> None:
        self.detectors: dict[str, Any] = {"step": _StepDetector()}

    def detect(
        self,
        window: Any,
        detector_types: list[str] | None = None,
        detector_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows = np.asarray(window, dtype=np.float64)
        scores = (rows[:, 0] > 1.0).astype(np.float64)
        return {
            "detectors": {
                "step": {"scores": scores.tolist(), "is_anomaly": (scores > 0.5).tolist()}
            }
        }


def _strict_json(value: Any) -> None:
    json.dumps(value, allow_nan=False)  # raises on NaN / Infinity


class TestEngineStreamSessionObserve:
    @settings(max_examples=250, deadline=None, suppress_health_check=_SLOW)
    @given(
        samples=st.lists(_point, min_size=1, max_size=40),
        monitor_drift=st.booleans(),
    )
    def test_arbitrary_samples_are_measured_unmeasured_or_refused(
        self, samples: list[Any], monitor_drift: bool
    ) -> None:
        session = EngineStreamSession(
            _StepEngine(),  # type: ignore[arg-type]
            detector_types=["step"],
            min_samples=4,
            window_size=16,
            refit_interval=5,
            monitor_drift=monitor_drift,
        )
        width: int | None = None
        unmeasured = 0
        for seen, sample in enumerate(samples, start=1):
            values = [sample] if isinstance(sample, float) else list(sample)
            measurable = bool(values) and all(math.isfinite(v) for v in values)
            if measurable and width is not None and len(values) != width:
                with pytest.raises(ValueError, match="features"):
                    session.observe(sample)
                continue
            verdict = session.observe(sample)  # must not raise
            _strict_json(verdict)
            assert verdict["samples_seen"] == seen
            if not measurable:
                unmeasured += 1
                # Unmeasured is reported; it is never scored, never an all-clear.
                assert verdict["status"] == "unmeasured"
                assert verdict["is_anomaly"] is None
                assert verdict["partial"] is True
            else:
                width = len(values)
                if verdict["status"] == "ok":
                    # The verdict is about THIS sample, whatever came before it.
                    assert verdict["is_anomaly"] is (values[0] > 1.0)
            assert verdict["samples_unmeasured"] == unmeasured


class TestHysteresisGate:
    @settings(max_examples=400, suppress_health_check=_SLOW)
    @given(
        exit_threshold=_finite,
        band=st.floats(min_value=0.0, max_value=1e3, allow_nan=False),
        scores=st.lists(_floats, max_size=60),
    )
    def test_the_last_decisive_score_decides(
        self, exit_threshold: float, band: float, scores: list[float]
    ) -> None:
        """Closed form: the verdict is set by the last finite score outside the band.

        A score above ``enter`` turns it on, one at or below ``exit`` turns it
        off, and anything else -- inside the band, NaN, infinite garbage that
        is not a number -- leaves it as it was.
        """
        enter_threshold = exit_threshold + band
        gate = HysteresisGate(enter_threshold, exit_threshold)
        expected = False
        for score in scores:
            before = gate.active
            verdict = gate.update(score)
            if not math.isfinite(score):
                assert verdict is before  # a failed reading moves nothing
            elif score > enter_threshold:
                expected = True
            elif score <= exit_threshold:
                expected = False
            assert verdict is expected
        batch = hysteresis_verdicts(
            np.asarray(scores, dtype=np.float64), enter_threshold, exit_threshold
        )
        assert batch.tolist() == _replay(scores, enter_threshold, exit_threshold)

    @settings(max_examples=200)
    @given(enter=_floats, leave=_floats)
    def test_a_band_is_refused_unless_finite_and_ordered(self, enter: float, leave: float) -> None:
        valid = math.isfinite(enter) and math.isfinite(leave) and leave <= enter
        if valid:
            assert HysteresisGate(enter, leave).active is False
        else:
            with pytest.raises(ValueError, match="threshold"):
                HysteresisGate(enter, leave)


def _replay(scores: list[float], enter: float, leave: float) -> list[bool]:
    """The streaming gate fed one score at a time (what the batch form must equal)."""
    gate = HysteresisGate(enter, leave)
    return [gate.update(score) for score in scores]
